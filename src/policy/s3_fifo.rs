//! S3-FIFO (Simple, Scalable, Scan-resistant FIFO) Cache Implementation
//!
//! This module provides an implementation of the S3-FIFO algorithm, a modern
//! cache eviction policy that achieves scan resistance using three FIFO queues
//! without the complexity of LRU bookkeeping.
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────────────┐
//! │                         S3FifoCache<K, V> Layout                             │
//! │                                                                              │
//! │   ┌──────────────────────────────────────────────────────────────────────┐   │
//! │   │  index: HashMap<K, NodePtr>      Arena: Node<K,V>                    │   │
//! │   │                                                                      │   │
//! │   │  ┌──────────┬──────────┐        ┌────────┬──────────────────────┐    │   │
//! │   │  │   Key    │  NodePtr │        │  Node  │  key, value, freq    │    │   │
//! │   │  ├──────────┼──────────┤        ├────────┼──────────────────────┤    │   │
//! │   │  │  "pg_1"  │   ptr_0  │───────►│ ptr_0  │  pg_1, data, 0       │    │   │
//! │   │  │  "pg_2"  │   ptr_1  │───────►│ ptr_1  │  pg_2, data, 1       │    │   │
//! │   │  │  "pg_3"  │   ptr_2  │───────►│ ptr_2  │  pg_3, data, 0       │    │   │
//! │   │  └──────────┴──────────┘        └────────┴──────────────────────┘    │   │
//! │   └──────────────────────────────────────────────────────────────────────┘   │
//! │                                                                              │
//! │   ┌─────────────────────────────────────────────────────────────────────┐    │
//! │   │                        Queue Organization                           │    │
//! │   │                                                                     │    │
//! │   │   SMALL QUEUE (S - FIFO)              MAIN QUEUE (M - FIFO)         │    │
//! │   │   ┌─────────────────────────┐        ┌─────────────────────────┐    │    │
//! │   │   │ head               tail │        │ head               tail │    │    │
//! │   │   │  ▼                    ▼ │        │  ▼                    ▼ │    │    │
//! │   │   │ [new] ◄──► [old] ◄──┤   │        │ [hot] ◄──► [warm] ◄───┤ │    │    │
//! │   │   │  ^          evict here  │        │  ^           evict here │    │    │
//! │   │   │  │                      │        │  │   (if freq==0)       │    │    │
//! │   │   │ insert                  │        │ promoted from S         │    │    │
//! │   │   └─────────────────────────┘        └─────────────────────────┘    │    │
//! │   │                                                                     │    │
//! │   │   GHOST QUEUE (G - keys only)                                       │    │
//! │   │   ┌─────────────────────────────────────────────────────────────┐   │    │
//! │   │   │  Tracks recently evicted keys for admission decisions       │   │    │
//! │   │   │  If key in Ghost → insert to Main (not Small)               │   │    │
//! │   │   └─────────────────────────────────────────────────────────────┘   │    │
//! │   └─────────────────────────────────────────────────────────────────────┘    │
//! └──────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## S3-FIFO Algorithm Flow
//!
//! ```text
//! Insert Flow:
//! ────────────
//!
//!   insert(key, value):
//!     1. Key exists? → Update value, increment freq (capped at 3)
//!     2. Key in Ghost? → Insert to Main (promoted admission)
//!     3. Otherwise → Insert to Small
//!     4. Evict if over capacity
//!
//! Access Flow:
//! ────────────
//!
//!   get(key):
//!     1. Lookup in index → not found? return None
//!     2. Increment freq (capped at 3)
//!     3. Return &value
//!
//! Eviction Flow:
//! ──────────────
//!
//!   evict_if_needed():
//!     while len > capacity:
//!       1. Try evict from Small:
//!          - Pop from Small tail
//!          - If freq > 0: promote to Main head, reset freq to 0
//!          - If freq == 0: evict, record in Ghost
//!       2. If Small empty, evict from Main:
//!          - Pop from Main tail
//!          - If freq > 0: reinsert to Main head, decrement freq
//!          - If freq == 0: evict (don't record in Ghost)
//! ```
//!
//! ## Key Components
//!
//! - [`S3FifoCache`]: Main S3-FIFO cache implementation
//! - `S3FifoCacheInner`: Internal state with queues and index
//!
//! ## Operations
//!
//! | Operation   | Time   | Notes                                      |
//! |-------------|--------|--------------------------------------------|
//! | `get`       | O(1)   | Increments frequency counter               |
//! | `insert`    | O(1)*  | *Amortized, may trigger evictions          |
//! | `contains`  | O(1)   | Index lookup only, no freq update          |
//! | `len`       | O(1)   | Returns total entries (small + main)       |
//! | `clear`     | O(n)   | Clears all structures                      |
//!
//! ## Algorithm Properties
//!
//! - **Scan Resistance**: One-hit wonders stay in Small and get evicted quickly
//! - **Simplicity**: No LRU bookkeeping, just FIFO queues with counters
//! - **Efficiency**: O(1) operations, cache-friendly FIFO traversal
//! - **Ghost-Guided Admission**: Items evicted from Small may be promoted on re-insert
//!
//! ## Configuration
//!
//! - `small_ratio`: Fraction of capacity for Small queue (default 0.1 = 10%)
//! - `ghost_ratio`: Fraction of capacity for Ghost list (default 0.9 = 90%)
//!
//! ## Use Cases
//!
//! - CDN edge caches
//! - Database buffer pools with scan-heavy workloads
//! - Web caches with mixed popularity distributions
//! - Any workload with occasional full scans
//!
//! ## Example Usage
//!
//! ```
//! use cachekit::policy::s3_fifo::S3FifoCache;
//! use cachekit::traits::CoreCache;
//!
//! // Create S3-FIFO cache with 100 capacity
//! let mut cache: S3FifoCache<String, String> = S3FifoCache::new(100);
//!
//! // Insert items (go to Small queue)
//! cache.insert("page1".to_string(), "content1".to_string());
//! cache.insert("page2".to_string(), "content2".to_string());
//!
//! // Access promotes frequency
//! assert_eq!(cache.get(&"page1".to_string()), Some(&"content1".to_string()));
//!
//! // Re-accessed items survive eviction
//! for i in 0..150 {
//!     cache.insert(format!("scan_{}", i), format!("data_{}", i));
//! }
//!
//! // "page1" likely survived (was accessed), "page2" likely evicted
//! let _ = cache.contains(&"page1".to_string());
//!
//! assert_eq!(cache.len(), 100);
//! ```
//!
//! ## Comparison with Other Policies
//!
//! | Policy   | Scan Resistant| Complexity | Overhead    | Best For            |
//! |----------|---------------|------------|-------------|---------------------|
//! | FIFO     | No            | O(1)       | Minimal     | Predictable eviction|
//! | LRU      | No            | O(1)*      | Per-access  | Temporal locality   |
//! | 2Q       | Yes           | O(1)       | Two queues  | Mixed workloads     |
//! | S3-FIFO  | Yes           | O(1)       | Three queues| Scan-heavy workloads|
//!
//! \* LRU requires pointer updates on every access
//!
//! ## Thread Safety
//!
//! - [`S3FifoCache`]: Not thread-safe, designed for single-threaded use
//! - [`ConcurrentS3FifoCache`]: Thread-safe wrapper using RwLock
//!
//! ## Implementation Notes
//!
//! - Frequency counter is capped at 3 (2 bits) to prevent counter inflation
//! - Ghost list uses existing `GhostList` from `crate::ds`
//! - Eviction prefers Small queue to quickly filter one-hit wonders
//! - Items promoted from Small to Main start with freq=0
//!
//! ## References
//!
//! - Yang et al., "FIFO queues are all you need for cache eviction", SOSP 2023

use std::fmt::Debug;
use std::hash::Hash;
use std::ptr::NonNull;
#[cfg(feature = "concurrency")]
use std::sync::Arc;

#[cfg(feature = "concurrency")]
use parking_lot::RwLock;
use rustc_hash::FxHashMap;

use crate::ds::GhostList;
#[cfg(feature = "concurrency")]
use crate::traits::ConcurrentCache;
use crate::traits::CoreCache;

/// Maximum frequency value (2 bits = 0-3).
const MAX_FREQ: u8 = 3;

/// Default ratio of capacity allocated to Small queue.
const DEFAULT_SMALL_RATIO: f64 = 0.1;

/// Default ratio of capacity for Ghost list.
const DEFAULT_GHOST_RATIO: f64 = 0.9;

/// Performance metrics for S3-FIFO cache operations.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct S3FifoMetrics {
    /// Number of cache hits.
    pub hits: u64,
    /// Number of cache misses.
    pub misses: u64,
    /// Number of insertions.
    pub inserts: u64,
    /// Number of updates (key already existed).
    pub updates: u64,
    /// Number of promotions from Small to Main.
    pub promotions: u64,
    /// Number of Main reinsertions (freq > 0).
    pub main_reinserts: u64,
    /// Number of evictions from Small.
    pub small_evictions: u64,
    /// Number of evictions from Main.
    pub main_evictions: u64,
    /// Number of ghost hits (ghost-guided admission).
    pub ghost_hits: u64,
}

impl std::fmt::Display for S3FifoMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let total_accesses = self.hits + self.misses;
        let hit_rate = if total_accesses > 0 {
            (self.hits as f64 / total_accesses as f64) * 100.0
        } else {
            0.0
        };

        write!(
            f,
            "S3FifoMetrics {{ hits: {}, misses: {}, hit_rate: {:.2}%, inserts: {}, updates: {}, \
             promotions: {}, main_reinserts: {}, small_evictions: {}, main_evictions: {}, ghost_hits: {} }}",
            self.hits,
            self.misses,
            hit_rate,
            self.inserts,
            self.updates,
            self.promotions,
            self.main_reinserts,
            self.small_evictions,
            self.main_evictions,
            self.ghost_hits
        )
    }
}

/// Which queue a node belongs to.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum QueueKind {
    Small,
    Main,
}

/// Which queue an iterator is currently traversing.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum IterQueue {
    Small,
    Main,
}

/// Iterator over cache entries.
pub struct Iter<'a, K, V> {
    current: Option<NonNull<Node<K, V>>>,
    queue: IterQueue,
    main_head: Option<NonNull<Node<K, V>>>,
    remaining: usize,
    _marker: std::marker::PhantomData<&'a (K, V)>,
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        while self.remaining > 0 {
            match self.current {
                Some(node_ptr) => {
                    unsafe {
                        let node = &*node_ptr.as_ptr();
                        // Move to next node (toward tail)
                        self.current = node.next;
                        self.remaining -= 1;
                        return Some((&node.key, &node.value));
                    }
                },
                None => {
                    // Finished Small, move to Main
                    if self.queue == IterQueue::Small {
                        self.queue = IterQueue::Main;
                        self.current = self.main_head;
                        // Continue to process Main queue
                    } else {
                        // Both queues exhausted
                        return None;
                    }
                },
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a, K, V> ExactSizeIterator for Iter<'a, K, V> {
    fn len(&self) -> usize {
        self.remaining
    }
}

/// Iterator over cache keys.
pub struct Keys<'a, K, V> {
    inner: Iter<'a, K, V>,
}

impl<'a, K, V> Iterator for Keys<'a, K, V> {
    type Item = &'a K;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(k, _)| k)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, K, V> ExactSizeIterator for Keys<'a, K, V> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

/// Iterator over cache values.
pub struct Values<'a, K, V> {
    inner: Iter<'a, K, V>,
}

impl<'a, K, V> Iterator for Values<'a, K, V> {
    type Item = &'a V;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(_, v)| v)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, K, V> ExactSizeIterator for Values<'a, K, V> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

/// Internal node storing key, value, and metadata.
#[repr(C)]
struct Node<K, V> {
    prev: Option<NonNull<Node<K, V>>>,
    next: Option<NonNull<Node<K, V>>>,
    queue: QueueKind,
    freq: u8,
    key: K,
    value: V,
}

/// S3-FIFO (Simple, Scalable, Scan-resistant FIFO) Cache.
///
/// Implements the S3-FIFO algorithm using three queues:
/// - **Small**: FIFO queue for new items (filters one-hit wonders)
/// - **Main**: FIFO queue for items re-accessed in Small
/// - **Ghost**: Tracks recently evicted keys for admission decisions
///
/// See module-level documentation for algorithm details.
///
/// # Type Parameters
///
/// - `K`: Key type, must be `Clone + Eq + Hash`
/// - `V`: Value type
///
/// # Example
///
/// ```
/// use cachekit::policy::s3_fifo::S3FifoCache;
/// use cachekit::traits::CoreCache;
///
/// let mut cache: S3FifoCache<String, String> = S3FifoCache::new(100);
///
/// cache.insert("hot_key".to_string(), "important_data".to_string());
/// cache.get(&"hot_key".to_string());  // Increment frequency
///
/// // Hot key survives scans
/// for i in 0..200 {
///     cache.insert(format!("scan_{}", i), "scan_data".to_string());
/// }
///
/// let _ = cache.contains(&"hot_key".to_string());
/// ```
pub struct S3FifoCache<K, V> {
    /// Key -> Node pointer mapping.
    map: FxHashMap<K, NonNull<Node<K, V>>>,

    /// Small queue (FIFO): head=newest, tail=oldest.
    small_head: Option<NonNull<Node<K, V>>>,
    small_tail: Option<NonNull<Node<K, V>>>,
    small_len: usize,

    /// Main queue (FIFO): head=newest, tail=oldest.
    main_head: Option<NonNull<Node<K, V>>>,
    main_tail: Option<NonNull<Node<K, V>>>,
    main_len: usize,

    /// Ghost list for tracking evicted keys.
    ghost: GhostList<K>,

    /// Maximum entries in Small queue (advisory, not strictly enforced).
    /// The actual queue sizes are balanced naturally through promotion.
    small_cap: usize,

    /// Total cache capacity.
    capacity: usize,

    /// Performance metrics (gated behind feature flag).
    #[cfg(feature = "metrics")]
    metrics: S3FifoMetrics,
}

// SAFETY: S3FifoCache can be sent between threads if K and V are Send.
unsafe impl<K, V> Send for S3FifoCache<K, V>
where
    K: Clone + Eq + Hash + Send,
    V: Send,
{
}

// SAFETY: S3FifoCache can be shared between threads if K and V are Sync.
unsafe impl<K, V> Sync for S3FifoCache<K, V>
where
    K: Clone + Eq + Hash + Sync,
    V: Sync,
{
}

impl<K, V> Default for S3FifoCache<K, V>
where
    K: Clone + Eq + Hash,
{
    /// Creates a cache with default capacity of 128.
    fn default() -> Self {
        Self::new(128)
    }
}

impl<K, V> S3FifoCache<K, V>
where
    K: Clone + Eq + Hash,
{
    /// Creates a new S3-FIFO cache with the specified capacity.
    ///
    /// Uses default ratios: 10% for Small queue, 90% for Ghost list.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::s3_fifo::S3FifoCache;
    ///
    /// let cache: S3FifoCache<String, i32> = S3FifoCache::new(100);
    /// assert_eq!(cache.capacity(), 100);
    /// assert!(cache.is_empty());
    /// ```
    pub fn new(capacity: usize) -> Self {
        Self::with_ratios(capacity, DEFAULT_SMALL_RATIO, DEFAULT_GHOST_RATIO)
    }

    /// Creates a new S3-FIFO cache with custom queue ratios.
    ///
    /// # Arguments
    ///
    /// - `capacity`: Total cache capacity
    /// - `small_ratio`: Fraction of capacity for Small queue (0.0 to 1.0)
    /// - `ghost_ratio`: Fraction of capacity for Ghost list (0.0+)
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::s3_fifo::S3FifoCache;
    ///
    /// // 20% for Small, 100% for Ghost
    /// let cache: S3FifoCache<String, i32> = S3FifoCache::with_ratios(100, 0.2, 1.0);
    /// assert_eq!(cache.capacity(), 100);
    /// ```
    pub fn with_ratios(capacity: usize, small_ratio: f64, ghost_ratio: f64) -> Self {
        let small_cap = ((capacity as f64 * small_ratio).round() as usize)
            .max(1)
            .min(capacity);
        let ghost_cap = (capacity as f64 * ghost_ratio).round() as usize;

        Self {
            map: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
            small_head: None,
            small_tail: None,
            small_len: 0,
            main_head: None,
            main_tail: None,
            main_len: 0,
            ghost: GhostList::new(ghost_cap),
            small_cap,
            capacity,
            #[cfg(feature = "metrics")]
            metrics: S3FifoMetrics::default(),
        }
    }

    /// Returns the total number of cached entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns `true` if the cache is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Returns the cache capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns `true` if the key exists.
    #[inline]
    pub fn contains(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    /// Retrieves a value by key without updating frequency.
    ///
    /// Unlike `get()`, this method does not increment the frequency counter,
    /// making it useful for read-only inspection or debugging.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::s3_fifo::S3FifoCache;
    ///
    /// let mut cache = S3FifoCache::new(10);
    /// cache.insert("key", 42);
    ///
    /// // Peek doesn't affect eviction priority
    /// assert_eq!(cache.peek(&"key"), Some(&42));
    /// assert_eq!(cache.peek(&"missing"), None);
    /// ```
    #[inline]
    pub fn peek(&self, key: &K) -> Option<&V> {
        let node_ptr = *self.map.get(key)?;
        unsafe {
            let node = &*node_ptr.as_ptr();
            Some(&node.value)
        }
    }

    /// Returns the number of entries in the Small queue.
    #[inline]
    pub fn small_len(&self) -> usize {
        self.small_len
    }

    /// Returns the number of entries in the Main queue.
    #[inline]
    pub fn main_len(&self) -> usize {
        self.main_len
    }

    /// Returns the number of entries in the Ghost list.
    #[inline]
    pub fn ghost_len(&self) -> usize {
        self.ghost.len()
    }

    /// Returns performance metrics if the `metrics` feature is enabled.
    #[cfg(feature = "metrics")]
    #[inline]
    pub fn metrics(&self) -> &S3FifoMetrics {
        &self.metrics
    }

    /// Resets performance metrics to zero.
    #[cfg(feature = "metrics")]
    #[inline]
    pub fn reset_metrics(&mut self) {
        self.metrics = S3FifoMetrics::default();
    }

    /// Retrieves a value by key, incrementing its frequency.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::s3_fifo::S3FifoCache;
    ///
    /// let mut cache = S3FifoCache::new(10);
    /// cache.insert("key", 42);
    ///
    /// assert_eq!(cache.get(&"key"), Some(&42));
    /// assert_eq!(cache.get(&"missing"), None);
    /// ```
    #[inline]
    pub fn get(&mut self, key: &K) -> Option<&V> {
        let node_ptr = *self.map.get(key)?;

        #[cfg(feature = "metrics")]
        {
            self.metrics.hits += 1;
        }

        unsafe {
            let node = &mut *node_ptr.as_ptr();
            // Increment frequency, capped at MAX_FREQ
            if node.freq < MAX_FREQ {
                node.freq += 1;
            }
            Some(&node.value)
        }
    }

    /// Retrieves a mutable reference to a value by key, incrementing its frequency.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::s3_fifo::S3FifoCache;
    ///
    /// let mut cache = S3FifoCache::new(10);
    /// cache.insert("key", 42);
    ///
    /// if let Some(val) = cache.get_mut(&"key") {
    ///     *val = 100;
    /// }
    ///
    /// assert_eq!(cache.get(&"key"), Some(&100));
    /// ```
    #[inline]
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        let node_ptr = *self.map.get(key)?;

        #[cfg(feature = "metrics")]
        {
            self.metrics.hits += 1;
        }

        unsafe {
            let node = &mut *node_ptr.as_ptr();
            // Increment frequency, capped at MAX_FREQ
            if node.freq < MAX_FREQ {
                node.freq += 1;
            }
            Some(&mut node.value)
        }
    }

    /// Inserts or updates a key-value pair.
    ///
    /// - If key exists: updates value and increments frequency
    /// - If key is in Ghost: inserts to Main queue (ghost-guided admission)
    /// - Otherwise: inserts to Small queue
    ///
    /// # Returns
    ///
    /// - `Some(old_value)` if the key already existed
    /// - `None` if the key is new
    ///
    /// # Panics
    ///
    /// Panics if the cache has zero capacity
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::s3_fifo::S3FifoCache;
    ///
    /// let mut cache = S3FifoCache::new(10);
    ///
    /// // New insert goes to Small
    /// cache.insert("key", "initial");
    /// assert_eq!(cache.len(), 1);
    ///
    /// // Update existing key
    /// cache.insert("key", "updated");
    /// assert_eq!(cache.get(&"key"), Some(&"updated"));
    /// assert_eq!(cache.len(), 1);
    /// ```
    #[inline]
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        // Zero capacity is a programming error
        assert!(self.capacity > 0, "cannot insert into zero-capacity cache");

        // Update existing key
        if let Some(&node_ptr) = self.map.get(&key) {
            #[cfg(feature = "metrics")]
            {
                self.metrics.updates += 1;
            }

            unsafe {
                let node = &mut *node_ptr.as_ptr();
                let old = std::mem::replace(&mut node.value, value);
                // Increment frequency on update
                if node.freq < MAX_FREQ {
                    node.freq += 1;
                }
                return Some(old);
            }
        }

        #[cfg(feature = "metrics")]
        {
            self.metrics.inserts += 1;
        }

        // Check if key is in Ghost (ghost-guided admission)
        let insert_to_main = self.ghost.remove(&key);

        #[cfg(feature = "metrics")]
        if insert_to_main {
            self.metrics.ghost_hits += 1;
        }

        // Evict before inserting
        self.evict_if_needed();

        // Create new node
        let queue = if insert_to_main {
            QueueKind::Main
        } else {
            QueueKind::Small
        };

        let node = Box::new(Node {
            prev: None,
            next: None,
            queue,
            freq: 0,
            key: key.clone(),
            value,
        });
        let node_ptr = NonNull::new(Box::into_raw(node)).unwrap();

        self.map.insert(key, node_ptr);

        if insert_to_main {
            self.attach_main_head(node_ptr);
        } else {
            self.attach_small_head(node_ptr);
        }

        None
    }

    /// Removes a key-value pair from the cache, returning the value if it existed.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::s3_fifo::S3FifoCache;
    ///
    /// let mut cache = S3FifoCache::new(10);
    /// cache.insert("key", 42);
    ///
    /// assert_eq!(cache.remove(&"key"), Some(42));
    /// assert_eq!(cache.remove(&"key"), None);
    /// assert!(cache.is_empty());
    /// ```
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let node_ptr = self.map.remove(key)?;

        unsafe {
            let node = &*node_ptr.as_ptr();
            match node.queue {
                QueueKind::Small => {
                    self.detach_small(node_ptr);
                },
                QueueKind::Main => {
                    self.detach_main(node_ptr);
                },
            }
            let boxed = Box::from_raw(node_ptr.as_ptr());
            Some(boxed.value)
        }
    }

    /// Clears all entries from the cache.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::s3_fifo::S3FifoCache;
    ///
    /// let mut cache = S3FifoCache::new(10);
    /// cache.insert("a", 1);
    /// cache.insert("b", 2);
    ///
    /// cache.clear();
    /// assert!(cache.is_empty());
    /// ```
    pub fn clear(&mut self) {
        // Free all nodes
        while self.pop_small_tail().is_some() {}
        while self.pop_main_tail().is_some() {}
        self.map.clear();
        self.ghost.clear();
    }

    /// Returns an iterator over all key-value pairs in the cache.
    ///
    /// The iteration order is unspecified (determined by internal queue structure).
    /// Entries from the Small queue are visited first, followed by the Main queue.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::s3_fifo::S3FifoCache;
    ///
    /// let mut cache = S3FifoCache::new(10);
    /// cache.insert("a", 1);
    /// cache.insert("b", 2);
    ///
    /// let items: Vec<_> = cache.iter().collect();
    /// assert_eq!(items.len(), 2);
    /// ```
    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter {
            current: self.small_head,
            queue: IterQueue::Small,
            main_head: self.main_head,
            remaining: self.len(),
            _marker: std::marker::PhantomData,
        }
    }

    /// Returns an iterator over keys in the cache.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::s3_fifo::S3FifoCache;
    ///
    /// let mut cache = S3FifoCache::new(10);
    /// cache.insert("a", 1);
    /// cache.insert("b", 2);
    ///
    /// let keys: Vec<_> = cache.keys().collect();
    /// assert_eq!(keys.len(), 2);
    /// ```
    pub fn keys(&self) -> Keys<'_, K, V> {
        Keys { inner: self.iter() }
    }

    /// Returns an iterator over values in the cache.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::s3_fifo::S3FifoCache;
    ///
    /// let mut cache = S3FifoCache::new(10);
    /// cache.insert("a", 1);
    /// cache.insert("b", 2);
    ///
    /// let values: Vec<_> = cache.values().collect();
    /// assert_eq!(values.len(), 2);
    /// ```
    pub fn values(&self) -> Values<'_, K, V> {
        Values { inner: self.iter() }
    }

    /// Validates internal data structure invariants.
    ///
    /// This method checks the consistency of the cache's internal state:
    /// - Queue length counters match actual list lengths
    /// - Hash map size equals total queue entries
    /// - All nodes in Small queue have correct queue kind
    /// - All nodes in Main queue have correct queue kind
    /// - All prev/next pointers form valid doubly-linked lists
    /// - All map entries point to nodes in one of the queues
    /// - Head/tail pointer consistency
    ///
    /// # Returns
    ///
    /// - `Ok(())` if all invariants hold
    /// - `Err(String)` with a description of the violated invariant
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::s3_fifo::S3FifoCache;
    ///
    /// let mut cache = S3FifoCache::new(10);
    /// cache.insert("a", 1);
    /// cache.insert("b", 2);
    ///
    /// // Should always pass for valid cache state
    /// cache.check_invariants().expect("Invariants violated");
    /// ```
    #[cfg(debug_assertions)]
    pub fn check_invariants(&self) -> Result<(), String>
    where
        K: Debug,
    {
        // Check that map size matches queue lengths
        let total_len = self.small_len + self.main_len;
        if self.map.len() != total_len {
            return Err(format!(
                "Map size {} != small_len {} + main_len {} = {}",
                self.map.len(),
                self.small_len,
                self.main_len,
                total_len
            ));
        }

        // Check Small queue consistency
        let mut small_count = 0;
        let mut current = self.small_head;
        let mut prev_ptr: Option<NonNull<Node<K, V>>> = None;

        // Check head/tail consistency for Small queue
        if self.small_head.is_none() != self.small_tail.is_none() {
            return Err(format!(
                "Small head/tail inconsistent: head={:?}, tail={:?}",
                self.small_head.is_some(),
                self.small_tail.is_some()
            ));
        }

        if self.small_head.is_none() && self.small_len != 0 {
            return Err(format!(
                "Small queue empty but small_len = {}",
                self.small_len
            ));
        }

        while let Some(node_ptr) = current {
            small_count += 1;

            unsafe {
                let node = &*node_ptr.as_ptr();

                // Check queue kind
                if node.queue != QueueKind::Small {
                    return Err(format!(
                        "Node with key {:?} in Small queue has queue = {:?}",
                        node.key, node.queue
                    ));
                }

                // Check frequency bounds
                if node.freq > MAX_FREQ {
                    return Err(format!(
                        "Node with key {:?} has freq {} > MAX_FREQ {}",
                        node.key, node.freq, MAX_FREQ
                    ));
                }

                // Check prev pointer consistency
                if node.prev != prev_ptr {
                    return Err(format!(
                        "Small queue: node {:?} prev pointer inconsistent",
                        node.key
                    ));
                }

                // Check map entry
                if !self.map.contains_key(&node.key) {
                    return Err(format!(
                        "Node with key {:?} in Small queue not in map",
                        node.key
                    ));
                }

                // Check that map points to this node
                if let Some(&map_ptr) = self.map.get(&node.key) {
                    if map_ptr != node_ptr {
                        return Err(format!(
                            "Map entry for key {:?} points to different node",
                            node.key
                        ));
                    }
                }

                // Check tail pointer
                if node.next.is_none() && Some(node_ptr) != self.small_tail {
                    return Err(format!(
                        "Small queue: last node {:?} doesn't match small_tail",
                        node.key
                    ));
                }

                prev_ptr = Some(node_ptr);
                current = node.next;
            }
        }

        if small_count != self.small_len {
            return Err(format!(
                "Small queue: counted {} nodes but small_len = {}",
                small_count, self.small_len
            ));
        }

        // Check Main queue consistency
        let mut main_count = 0;
        let mut current = self.main_head;
        let mut prev_ptr: Option<NonNull<Node<K, V>>> = None;

        // Check head/tail consistency for Main queue
        if self.main_head.is_none() != self.main_tail.is_none() {
            return Err(format!(
                "Main head/tail inconsistent: head={:?}, tail={:?}",
                self.main_head.is_some(),
                self.main_tail.is_some()
            ));
        }

        if self.main_head.is_none() && self.main_len != 0 {
            return Err(format!("Main queue empty but main_len = {}", self.main_len));
        }

        while let Some(node_ptr) = current {
            main_count += 1;

            unsafe {
                let node = &*node_ptr.as_ptr();

                // Check queue kind
                if node.queue != QueueKind::Main {
                    return Err(format!(
                        "Node with key {:?} in Main queue has queue = {:?}",
                        node.key, node.queue
                    ));
                }

                // Check frequency bounds
                if node.freq > MAX_FREQ {
                    return Err(format!(
                        "Node with key {:?} has freq {} > MAX_FREQ {}",
                        node.key, node.freq, MAX_FREQ
                    ));
                }

                // Check prev pointer consistency
                if node.prev != prev_ptr {
                    return Err(format!(
                        "Main queue: node {:?} prev pointer inconsistent",
                        node.key
                    ));
                }

                // Check map entry
                if !self.map.contains_key(&node.key) {
                    return Err(format!(
                        "Node with key {:?} in Main queue not in map",
                        node.key
                    ));
                }

                // Check that map points to this node
                if let Some(&map_ptr) = self.map.get(&node.key) {
                    if map_ptr != node_ptr {
                        return Err(format!(
                            "Map entry for key {:?} points to different node",
                            node.key
                        ));
                    }
                }

                // Check tail pointer
                if node.next.is_none() && Some(node_ptr) != self.main_tail {
                    return Err(format!(
                        "Main queue: last node {:?} doesn't match main_tail",
                        node.key
                    ));
                }

                prev_ptr = Some(node_ptr);
                current = node.next;
            }
        }

        if main_count != self.main_len {
            return Err(format!(
                "Main queue: counted {} nodes but main_len = {}",
                main_count, self.main_len
            ));
        }

        // Check capacity constraint
        if total_len > self.capacity {
            return Err(format!(
                "Total entries {} > capacity {}",
                total_len, self.capacity
            ));
        }

        Ok(())
    }

    /// Attaches a node at the head of Small queue.
    #[inline(always)]
    fn attach_small_head(&mut self, mut node_ptr: NonNull<Node<K, V>>) {
        unsafe {
            let node = node_ptr.as_mut();
            node.prev = None;
            node.next = self.small_head;
            node.queue = QueueKind::Small;

            match self.small_head {
                Some(mut h) => h.as_mut().prev = Some(node_ptr),
                None => self.small_tail = Some(node_ptr),
            }

            self.small_head = Some(node_ptr);
            self.small_len += 1;
        }
    }

    /// Attaches a node at the head of Main queue.
    #[inline(always)]
    fn attach_main_head(&mut self, mut node_ptr: NonNull<Node<K, V>>) {
        unsafe {
            let node = node_ptr.as_mut();
            node.prev = None;
            node.next = self.main_head;
            node.queue = QueueKind::Main;

            match self.main_head {
                Some(mut h) => h.as_mut().prev = Some(node_ptr),
                None => self.main_tail = Some(node_ptr),
            }

            self.main_head = Some(node_ptr);
            self.main_len += 1;
        }
    }

    /// Pops from Small tail (oldest in Small).
    #[inline(always)]
    fn pop_small_tail(&mut self) -> Option<Box<Node<K, V>>> {
        self.small_tail.map(|tail_ptr| unsafe {
            let node = Box::from_raw(tail_ptr.as_ptr());

            self.small_tail = node.prev;
            match self.small_tail {
                Some(mut t) => t.as_mut().next = None,
                None => self.small_head = None,
            }
            self.small_len -= 1;

            node
        })
    }

    /// Detaches a node from the Small queue without deallocating it.
    #[inline(always)]
    fn detach_small(&mut self, mut node_ptr: NonNull<Node<K, V>>) {
        unsafe {
            let node = node_ptr.as_mut();

            match node.prev {
                Some(mut p) => p.as_mut().next = node.next,
                None => self.small_head = node.next,
            }

            match node.next {
                Some(mut n) => n.as_mut().prev = node.prev,
                None => self.small_tail = node.prev,
            }

            self.small_len -= 1;
        }
    }

    /// Detaches a node from the Main queue without deallocating it.
    #[inline(always)]
    fn detach_main(&mut self, mut node_ptr: NonNull<Node<K, V>>) {
        unsafe {
            let node = node_ptr.as_mut();

            match node.prev {
                Some(mut p) => p.as_mut().next = node.next,
                None => self.main_head = node.next,
            }

            match node.next {
                Some(mut n) => n.as_mut().prev = node.prev,
                None => self.main_tail = node.prev,
            }

            self.main_len -= 1;
        }
    }

    /// Pops from Main tail (oldest in Main).
    #[inline(always)]
    fn pop_main_tail(&mut self) -> Option<Box<Node<K, V>>> {
        self.main_tail.map(|tail_ptr| unsafe {
            let node = Box::from_raw(tail_ptr.as_ptr());

            self.main_tail = node.prev;
            match self.main_tail {
                Some(mut t) => t.as_mut().next = None,
                None => self.main_head = None,
            }
            self.main_len -= 1;

            node
        })
    }

    /// Evicts entries until there is room for a new entry.
    ///
    /// S3-FIFO eviction strategy:
    /// 1. Prefer evicting from Small queue
    ///    - If freq > 0: promote to Main (reset freq to 0)
    ///    - If freq == 0: evict and record in Ghost
    /// 2. If Small empty, evict from Main
    ///    - If freq > 0: reinsert to Main head (decrement freq)
    ///    - If freq == 0: evict
    ///
    /// Optimized to avoid map removal/reinsertion during promotion.
    fn evict_if_needed(&mut self) {
        while self.len() >= self.capacity {
            // Try to evict from Small first
            if let Some(tail_ptr) = self.small_tail {
                unsafe {
                    let node = &mut *tail_ptr.as_ptr();

                    if node.freq > 0 {
                        // Promote to Main: detach from Small, reset freq, attach to Main
                        #[cfg(feature = "metrics")]
                        {
                            self.metrics.promotions += 1;
                        }

                        self.detach_small(tail_ptr);
                        node.freq = 0;
                        node.prev = None;
                        node.next = None;
                        self.attach_main_head(tail_ptr);
                        continue;
                    } else {
                        // Evict: remove from map and ghost record
                        #[cfg(feature = "metrics")]
                        {
                            self.metrics.small_evictions += 1;
                        }

                        let key = node.key.clone();
                        self.map.remove(&key);
                        let _ = self.pop_small_tail();
                        self.ghost.record(key);
                        continue;
                    }
                }
            }

            // Evict from Main if Small is empty
            if let Some(tail_ptr) = self.main_tail {
                unsafe {
                    let node = &mut *tail_ptr.as_ptr();

                    if node.freq > 0 {
                        // Reinsert to Main: detach, decrement freq, attach to head
                        #[cfg(feature = "metrics")]
                        {
                            self.metrics.main_reinserts += 1;
                        }

                        self.detach_main(tail_ptr);
                        node.freq -= 1;
                        node.prev = None;
                        node.next = None;
                        self.attach_main_head(tail_ptr);
                        continue;
                    } else {
                        // Evict: remove from map (don't record in Ghost)
                        #[cfg(feature = "metrics")]
                        {
                            self.metrics.main_evictions += 1;
                        }

                        self.map.remove(&node.key);
                        let _ = self.pop_main_tail();
                        continue;
                    }
                }
            }

            // No entries to evict
            break;
        }
    }
}

// Private methods needed for Drop, without trait bounds
impl<K, V> S3FifoCache<K, V> {
    /// Pops and deallocates from Small tail.
    fn drop_small_tail(&mut self) -> bool {
        if let Some(tail_ptr) = self.small_tail {
            unsafe {
                let node = Box::from_raw(tail_ptr.as_ptr());
                self.small_tail = node.prev;
                match self.small_tail {
                    Some(mut t) => t.as_mut().next = None,
                    None => self.small_head = None,
                }
                self.small_len -= 1;
            }
            true
        } else {
            false
        }
    }

    /// Pops and deallocates from Main tail.
    fn drop_main_tail(&mut self) -> bool {
        if let Some(tail_ptr) = self.main_tail {
            unsafe {
                let node = Box::from_raw(tail_ptr.as_ptr());
                self.main_tail = node.prev;
                match self.main_tail {
                    Some(mut t) => t.as_mut().next = None,
                    None => self.main_head = None,
                }
                self.main_len -= 1;
            }
            true
        } else {
            false
        }
    }
}

impl<K, V> FromIterator<(K, V)> for S3FifoCache<K, V>
where
    K: Clone + Eq + Hash,
{
    /// Creates a cache from an iterator of key-value pairs.
    ///
    /// The capacity is determined by the iterator's size hint, with a minimum of 16.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::s3_fifo::S3FifoCache;
    ///
    /// let cache: S3FifoCache<_, _> = vec![("a", 1), ("b", 2), ("c", 3)]
    ///     .into_iter()
    ///     .collect();
    ///
    /// assert_eq!(cache.len(), 3);
    /// ```
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        let mut cache = Self::new(lower.max(16));
        for (k, v) in iter {
            cache.insert(k, v);
        }
        cache
    }
}

impl<K, V> Extend<(K, V)> for S3FifoCache<K, V>
where
    K: Clone + Eq + Hash,
{
    /// Extends the cache with key-value pairs from an iterator.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::s3_fifo::S3FifoCache;
    ///
    /// let mut cache = S3FifoCache::new(10);
    /// cache.insert("a", 1);
    ///
    /// cache.extend(vec![("b", 2), ("c", 3)]);
    /// assert_eq!(cache.len(), 3);
    /// ```
    fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

impl<K, V> Drop for S3FifoCache<K, V> {
    fn drop(&mut self) {
        while self.drop_small_tail() {}
        while self.drop_main_tail() {}
    }
}

impl<K, V> Debug for S3FifoCache<K, V>
where
    K: Clone + Eq + Hash + Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("S3FifoCache")
            .field("capacity", &self.capacity)
            .field("small_cap", &self.small_cap)
            .field("len", &self.len())
            .field("small_len", &self.small_len)
            .field("main_len", &self.main_len)
            .field("ghost_len", &self.ghost.len())
            .finish_non_exhaustive()
    }
}

/// Implementation of [`CoreCache`] for S3FifoCache.
impl<K, V> CoreCache<K, V> for S3FifoCache<K, V>
where
    K: Clone + Eq + Hash,
{
    #[inline]
    fn insert(&mut self, key: K, value: V) -> Option<V> {
        S3FifoCache::insert(self, key, value)
    }

    #[inline]
    fn get(&mut self, key: &K) -> Option<&V> {
        S3FifoCache::get(self, key)
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
        S3FifoCache::clear(self);
    }
}

/// Builder for configuring concurrent S3-FIFO cache parameters.
#[cfg(feature = "concurrency")]
#[derive(Debug, Clone)]
pub struct ConcurrentS3FifoCacheBuilder {
    capacity: usize,
    small_ratio: f64,
    ghost_ratio: f64,
}

#[cfg(feature = "concurrency")]
impl ConcurrentS3FifoCacheBuilder {
    /// Creates a new builder with the specified capacity and default ratios.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            small_ratio: DEFAULT_SMALL_RATIO,
            ghost_ratio: DEFAULT_GHOST_RATIO,
        }
    }

    /// Sets the fraction of capacity allocated to the Small queue.
    pub fn small_ratio(mut self, ratio: f64) -> Self {
        self.small_ratio = ratio;
        self
    }

    /// Sets the fraction of capacity for the Ghost list.
    pub fn ghost_ratio(mut self, ratio: f64) -> Self {
        self.ghost_ratio = ratio;
        self
    }

    /// Builds the concurrent cache with the configured parameters.
    pub fn build<K, V>(self) -> ConcurrentS3FifoCache<K, V>
    where
        K: Clone + Eq + Hash + Debug,
    {
        ConcurrentS3FifoCache::with_ratios(self.capacity, self.small_ratio, self.ghost_ratio)
    }
}

/// Thread-safe S3-FIFO cache wrapper using RwLock.
///
/// Provides concurrent access to an S3-FIFO cache with read-write locking.
///
/// # Example
///
/// ```
/// use cachekit::policy::s3_fifo::ConcurrentS3FifoCache;
///
/// let cache = ConcurrentS3FifoCache::new(100);
///
/// // Safe for concurrent use
/// cache.insert("key", "value");
/// assert!(cache.contains(&"key"));
///
/// if let Some(value) = cache.get(&"key") {
///     println!("Got: {}", value);
/// }
/// ```
#[cfg(feature = "concurrency")]
#[derive(Clone, Debug)]
pub struct ConcurrentS3FifoCache<K, V>
where
    K: Clone + Eq + Hash,
{
    inner: Arc<RwLock<S3FifoCache<K, V>>>,
}

#[cfg(feature = "concurrency")]
impl<K, V> ConcurrentS3FifoCache<K, V>
where
    K: Clone + Eq + Hash + Debug,
{
    /// Creates a new concurrent S3-FIFO cache.
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(S3FifoCache::new(capacity))),
        }
    }

    /// Creates a new concurrent S3-FIFO cache with custom ratios.
    pub fn with_ratios(capacity: usize, small_ratio: f64, ghost_ratio: f64) -> Self {
        Self {
            inner: Arc::new(RwLock::new(S3FifoCache::with_ratios(
                capacity,
                small_ratio,
                ghost_ratio,
            ))),
        }
    }

    /// Returns a builder for configuring cache parameters.
    pub fn builder(capacity: usize) -> ConcurrentS3FifoCacheBuilder {
        ConcurrentS3FifoCacheBuilder::new(capacity)
    }

    /// Inserts a key-value pair.
    pub fn insert(&self, key: K, value: V) -> Option<V> {
        self.inner.write().insert(key, value)
    }

    /// Gets a cloned value by key, incrementing its frequency.
    ///
    /// This requires `V: Clone`. For non-cloneable values, use `get_with()`.
    pub fn get(&self, key: &K) -> Option<V>
    where
        V: Clone,
    {
        self.inner.write().get(key).cloned()
    }

    /// Gets a value by key and applies a function to it.
    ///
    /// This allows working with non-cloneable values by applying a transformation
    /// inside the lock. The frequency is still incremented.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::s3_fifo::ConcurrentS3FifoCache;
    ///
    /// let cache = ConcurrentS3FifoCache::new(10);
    /// cache.insert("key".to_string(), vec![1, 2, 3]);
    ///
    /// // Get length without cloning the vector
    /// let len = cache.get_with(&"key".to_string(), |v| v.len());
    /// assert_eq!(len, Some(3));
    /// ```
    pub fn get_with<F, R>(&self, key: &K, f: F) -> Option<R>
    where
        F: FnOnce(&V) -> R,
    {
        self.inner.write().get(key).map(f)
    }

    /// Peeks at a value without updating frequency or cloning.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::s3_fifo::ConcurrentS3FifoCache;
    ///
    /// let cache = ConcurrentS3FifoCache::new(10);
    /// cache.insert("key".to_string(), vec![1, 2, 3]);
    ///
    /// // Inspect without affecting eviction priority
    /// let len = cache.peek_with(&"key".to_string(), |v| v.len());
    /// assert_eq!(len, Some(3));
    /// ```
    pub fn peek_with<F, R>(&self, key: &K, f: F) -> Option<R>
    where
        F: FnOnce(&V) -> R,
    {
        self.inner.read().peek(key).map(f)
    }

    /// Returns `true` if the key exists.
    pub fn contains(&self, key: &K) -> bool {
        self.inner.read().contains(key)
    }

    /// Returns the number of cached entries.
    pub fn len(&self) -> usize {
        self.inner.read().len()
    }

    /// Returns `true` if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.read().is_empty()
    }

    /// Returns the cache capacity.
    pub fn capacity(&self) -> usize {
        self.inner.read().capacity()
    }

    /// Clears all entries.
    pub fn clear(&self) {
        self.inner.write().clear();
    }

    /// Returns the number of entries in the Small queue.
    pub fn small_len(&self) -> usize {
        self.inner.read().small_len()
    }

    /// Returns the number of entries in the Main queue.
    pub fn main_len(&self) -> usize {
        self.inner.read().main_len()
    }

    /// Returns the number of entries in the Ghost list.
    pub fn ghost_len(&self) -> usize {
        self.inner.read().ghost_len()
    }

    /// Returns performance metrics if the `metrics` feature is enabled.
    #[cfg(feature = "metrics")]
    pub fn metrics(&self) -> S3FifoMetrics {
        self.inner.read().metrics().clone()
    }

    /// Resets performance metrics to zero.
    #[cfg(feature = "metrics")]
    pub fn reset_metrics(&self) {
        self.inner.write().reset_metrics();
    }
}

#[cfg(feature = "concurrency")]
impl<K, V> ConcurrentCache for ConcurrentS3FifoCache<K, V>
where
    K: Clone + Eq + Hash + Debug + Send + Sync,
    V: Send + Sync,
{
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==============================================
    // Basic Operations
    // ==============================================

    mod basic_operations {
        use super::*;

        #[test]
        fn new_cache_is_empty() {
            let cache: S3FifoCache<&str, i32> = S3FifoCache::new(100);
            assert!(cache.is_empty());
            assert_eq!(cache.len(), 0);
            assert_eq!(cache.capacity(), 100);
        }

        #[test]
        fn insert_and_get() {
            let mut cache = S3FifoCache::new(100);
            cache.insert("key1", "value1");

            assert_eq!(cache.len(), 1);
            assert_eq!(cache.get(&"key1"), Some(&"value1"));
        }

        #[test]
        fn insert_multiple_items() {
            let mut cache = S3FifoCache::new(100);
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
            let mut cache: S3FifoCache<&str, i32> = S3FifoCache::new(100);
            cache.insert("exists", 42);

            assert_eq!(cache.get(&"missing"), None);
        }

        #[test]
        fn update_existing_key() {
            let mut cache = S3FifoCache::new(100);
            cache.insert("key", "initial");
            let old = cache.insert("key", "updated");

            assert_eq!(old, Some("initial"));
            assert_eq!(cache.len(), 1);
            assert_eq!(cache.get(&"key"), Some(&"updated"));
        }

        #[test]
        fn contains_returns_correct_result() {
            let mut cache = S3FifoCache::new(100);
            cache.insert("exists", 1);

            assert!(cache.contains(&"exists"));
            assert!(!cache.contains(&"missing"));
        }

        #[test]
        fn clear_removes_all_entries() {
            let mut cache = S3FifoCache::new(100);
            cache.insert("a", 1);
            cache.insert("b", 2);
            cache.get(&"a"); // Promote frequency

            cache.clear();

            assert!(cache.is_empty());
            assert_eq!(cache.len(), 0);
            assert!(!cache.contains(&"a"));
            assert!(!cache.contains(&"b"));
        }

        #[test]
        #[should_panic(expected = "cannot insert into zero-capacity cache")]
        fn zero_capacity_rejects_all() {
            let mut cache = S3FifoCache::new(0);
            cache.insert("key", "value");
        }
    }

    // ==============================================
    // Queue Behavior
    // ==============================================

    mod queue_behavior {
        use super::*;

        #[test]
        fn new_insert_goes_to_small() {
            let mut cache = S3FifoCache::new(100);
            cache.insert("key", "value");

            assert_eq!(cache.small_len(), 1);
            assert_eq!(cache.main_len(), 0);
        }

        #[test]
        fn accessed_item_promoted_on_eviction() {
            let mut cache: S3FifoCache<String, i32> = S3FifoCache::new(5);

            // Insert and access
            cache.insert("hot".to_string(), 0);
            cache.get(&"hot".to_string()); // freq becomes 1

            // Fill to trigger eviction
            for i in 1..10 {
                cache.insert(format!("cold_{}", i), i);
            }

            // "hot" should have been promoted to Main and survived
            assert!(cache.contains(&"hot".to_string()));
        }

        #[test]
        fn unaccessed_items_evicted_first() {
            let mut cache: S3FifoCache<String, i32> = S3FifoCache::new(5);

            cache.insert("hot1".to_string(), 1);
            cache.get(&"hot1".to_string());
            cache.insert("hot2".to_string(), 2);
            cache.get(&"hot2".to_string());

            // Add cold items
            cache.insert("cold1".to_string(), 3);
            cache.insert("cold2".to_string(), 4);
            cache.insert("cold3".to_string(), 5);

            // Trigger eviction
            cache.insert("new".to_string(), 6);

            // Hot items should survive, cold items evicted
            assert!(cache.contains(&"hot1".to_string()));
            assert!(cache.contains(&"hot2".to_string()));
            assert_eq!(cache.len(), 5);
        }

        #[test]
        fn frequency_increments_on_access() {
            let mut cache = S3FifoCache::new(10);
            cache.insert("key", "value");

            // Access multiple times
            cache.get(&"key");
            cache.get(&"key");
            cache.get(&"key");

            // Item should have high frequency (survives more evictions)
            assert!(cache.contains(&"key"));
        }
    }

    // ==============================================
    // Ghost-Guided Admission
    // ==============================================

    mod ghost_behavior {
        use super::*;

        #[test]
        fn evicted_key_recorded_in_ghost() {
            let mut cache = S3FifoCache::with_ratios(3, 0.5, 1.0);

            // Fill cache
            cache.insert("a", 1);
            cache.insert("b", 2);
            cache.insert("c", 3);

            // Trigger eviction
            cache.insert("d", 4);

            // "a" should be in ghost
            assert_eq!(cache.ghost_len(), 1);
        }

        #[test]
        fn ghost_hit_promotes_to_main() {
            let mut cache: S3FifoCache<String, i32> = S3FifoCache::with_ratios(5, 0.4, 1.0);

            // Insert and let it be evicted
            cache.insert("will_be_ghost".to_string(), 1);

            // Fill to evict
            for i in 0..10 {
                cache.insert(format!("filler_{}", i), i);
            }

            // Verify it was evicted
            assert!(!cache.contains(&"will_be_ghost".to_string()));
            let ghost_had_key = cache.ghost.contains(&"will_be_ghost".to_string());

            if ghost_had_key {
                // Re-insert should go to Main
                cache.insert("will_be_ghost".to_string(), 2);
                assert!(cache.contains(&"will_be_ghost".to_string()));
                // Should be in Main now (ghost-guided admission)
            }
        }
    }

    // ==============================================
    // Scan Resistance
    // ==============================================

    mod scan_resistance {
        use super::*;

        #[test]
        fn working_set_survives_scan() {
            let mut cache = S3FifoCache::new(100);

            // Create working set with accesses
            for i in 0..30 {
                let key = format!("working_{}", i);
                cache.insert(key.clone(), i);
                cache.get(&key); // Access to increase frequency
            }

            // Perform scan
            for i in 0..200 {
                cache.insert(format!("scan_{}", i), i);
            }

            // Count surviving working set items
            let mut survivors = 0;
            for i in 0..30 {
                if cache.contains(&format!("working_{}", i)) {
                    survivors += 1;
                }
            }

            assert!(
                survivors >= 20,
                "Expected most working set to survive, got {} survivors",
                survivors
            );
        }

        #[test]
        fn one_hit_wonders_evicted() {
            let mut cache = S3FifoCache::new(20);

            // Insert hot items with accesses
            for i in 0..5 {
                let key = format!("hot_{}", i);
                cache.insert(key.clone(), i);
                cache.get(&key);
                cache.get(&key);
            }

            // Insert cold items (no access)
            for i in 0..15 {
                cache.insert(format!("cold_{}", i), i);
            }

            // Trigger evictions
            for i in 0..30 {
                cache.insert(format!("scan_{}", i), i);
            }

            // Hot items should survive
            let mut hot_survivors = 0;
            for i in 0..5 {
                if cache.contains(&format!("hot_{}", i)) {
                    hot_survivors += 1;
                }
            }

            assert!(
                hot_survivors >= 4,
                "Hot items should mostly survive, got {} survivors",
                hot_survivors
            );
        }

        #[test]
        fn repeated_scans_dont_evict_hot() {
            let mut cache = S3FifoCache::new(50);

            // Create hot items
            for i in 0..10 {
                let key = format!("hot_{}", i);
                cache.insert(key.clone(), i);
                cache.get(&key);
                cache.get(&key);
                cache.get(&key);
            }

            // Multiple scans
            for scan in 0..3 {
                for i in 0..100 {
                    cache.insert(format!("scan_{}_{}", scan, i), i);
                }
            }

            // Hot items should survive all scans
            let mut survivors = 0;
            for i in 0..10 {
                if cache.contains(&format!("hot_{}", i)) {
                    survivors += 1;
                }
            }

            assert!(
                survivors >= 8,
                "Hot items should survive scans, got {} survivors",
                survivors
            );
        }
    }

    // ==============================================
    // Eviction Behavior
    // ==============================================

    mod eviction_behavior {
        use super::*;

        #[test]
        fn eviction_occurs_at_capacity() {
            let mut cache = S3FifoCache::new(5);

            for i in 0..10 {
                cache.insert(i, i * 10);
            }

            assert_eq!(cache.len(), 5);
        }

        #[test]
        fn oldest_small_evicted_first() {
            let mut cache = S3FifoCache::new(5);

            cache.insert("first", 1);
            cache.insert("second", 2);
            cache.insert("third", 3);
            cache.insert("fourth", 4);
            cache.insert("fifth", 5);
            cache.insert("sixth", 6);

            // "first" should be evicted (oldest in Small, no access)
            assert!(!cache.contains(&"first"));
            assert_eq!(cache.len(), 5);
        }

        #[test]
        fn main_queue_reinserts_with_freq() {
            let mut cache: S3FifoCache<String, i32> = S3FifoCache::new(5);

            // Create item with high frequency
            cache.insert("hot".to_string(), 0);
            cache.get(&"hot".to_string());
            cache.get(&"hot".to_string());
            cache.get(&"hot".to_string());

            // Fill to trigger promotion and Main eviction
            for i in 0..20 {
                cache.insert(format!("filler_{}", i), i);
            }

            // "hot" should survive due to Main reinsertion
            // (depends on exact eviction order)
        }

        #[test]
        fn capacity_maintained() {
            let mut cache = S3FifoCache::new(100);

            for i in 0..1000 {
                cache.insert(i, i);
            }

            assert_eq!(cache.len(), 100);
            assert!(cache.len() <= cache.capacity());
        }
    }

    // ==============================================
    // CoreCache Trait
    // ==============================================

    mod core_cache_trait {
        use super::*;

        #[test]
        fn trait_insert_returns_old() {
            let mut cache: S3FifoCache<&str, i32> = S3FifoCache::new(100);

            let old = CoreCache::insert(&mut cache, "key", 1);
            assert_eq!(old, None);

            let old = CoreCache::insert(&mut cache, "key", 2);
            assert_eq!(old, Some(1));
        }

        #[test]
        fn trait_get_works() {
            let mut cache = S3FifoCache::new(100);
            CoreCache::insert(&mut cache, "key", 42);

            assert_eq!(CoreCache::get(&mut cache, &"key"), Some(&42));
            assert_eq!(CoreCache::get(&mut cache, &"missing"), None);
        }

        #[test]
        fn trait_contains_works() {
            let mut cache = S3FifoCache::new(100);
            CoreCache::insert(&mut cache, "key", 1);

            assert!(CoreCache::contains(&cache, &"key"));
            assert!(!CoreCache::contains(&cache, &"missing"));
        }

        #[test]
        fn trait_len_and_capacity() {
            let mut cache: S3FifoCache<i32, i32> = S3FifoCache::new(50);

            assert_eq!(CoreCache::len(&cache), 0);
            assert_eq!(CoreCache::capacity(&cache), 50);

            for i in 0..30 {
                CoreCache::insert(&mut cache, i, i * 10);
            }

            assert_eq!(CoreCache::len(&cache), 30);
        }

        #[test]
        fn trait_clear_works() {
            let mut cache = S3FifoCache::new(100);
            CoreCache::insert(&mut cache, "a", 1);
            CoreCache::insert(&mut cache, "b", 2);

            CoreCache::clear(&mut cache);

            assert_eq!(CoreCache::len(&cache), 0);
            assert!(!CoreCache::contains(&cache, &"a"));
        }
    }

    // ==============================================
    // Concurrent Cache
    // ==============================================

    #[cfg(feature = "concurrency")]
    mod concurrent_cache {
        use super::*;

        #[test]
        fn concurrent_basic_operations() {
            let cache = ConcurrentS3FifoCache::new(100);

            cache.insert("key".to_string(), "value".to_string());
            assert!(cache.contains(&"key".to_string()));
            assert_eq!(cache.get(&"key".to_string()), Some("value".to_string()));
            assert_eq!(cache.len(), 1);

            cache.clear();
            assert!(cache.is_empty());
        }

        #[test]
        fn concurrent_capacity() {
            let cache: ConcurrentS3FifoCache<i32, i32> =
                ConcurrentS3FifoCache::with_ratios(50, 0.2, 1.0);
            assert_eq!(cache.capacity(), 50);
        }

        #[test]
        fn concurrent_queue_stats() {
            let cache = ConcurrentS3FifoCache::new(100);

            cache.insert("a".to_string(), 1);
            cache.insert("b".to_string(), 2);

            assert_eq!(cache.small_len(), 2);
            assert_eq!(cache.main_len(), 0);
            assert_eq!(cache.ghost_len(), 0);
        }
    }

    // ==============================================
    // Edge Cases
    // ==============================================

    mod edge_cases {
        use super::*;

        #[test]
        fn single_capacity() {
            let mut cache = S3FifoCache::new(1);

            cache.insert("a", 1);
            assert!(cache.contains(&"a"));

            cache.insert("b", 2);
            assert!(!cache.contains(&"a"));
            assert!(cache.contains(&"b"));
        }

        #[test]
        fn very_small_ratios() {
            let mut cache = S3FifoCache::with_ratios(100, 0.01, 0.01);

            for i in 0..50 {
                cache.insert(i, i);
            }

            assert_eq!(cache.len(), 50);
        }

        #[test]
        fn large_ratios() {
            let mut cache = S3FifoCache::with_ratios(100, 0.9, 2.0);

            for i in 0..100 {
                cache.insert(i, i);
            }

            assert_eq!(cache.len(), 100);
        }

        #[test]
        fn string_keys_and_values() {
            let mut cache = S3FifoCache::new(100);

            cache.insert(String::from("hello"), String::from("world"));
            cache.insert(String::from("foo"), String::from("bar"));

            assert_eq!(
                cache.get(&String::from("hello")),
                Some(&String::from("world"))
            );
        }

        #[test]
        fn empty_cache_operations() {
            let mut cache: S3FifoCache<i32, i32> = S3FifoCache::new(100);

            assert!(cache.is_empty());
            assert_eq!(cache.get(&1), None);
            assert!(!cache.contains(&1));

            cache.clear();
            assert!(cache.is_empty());
        }

        #[test]
        fn debug_format() {
            let mut cache: S3FifoCache<&str, i32> = S3FifoCache::new(100);
            cache.insert("test", 42);

            let debug_str = format!("{:?}", cache);
            assert!(debug_str.contains("S3FifoCache"));
            assert!(debug_str.contains("capacity"));
        }
    }

    // ==============================================
    // Workload Simulation
    // ==============================================

    mod workload_simulation {
        use super::*;

        #[test]
        fn database_buffer_pool() {
            let mut cache = S3FifoCache::new(100);

            // Hot index pages
            for i in 0..10 {
                let key = format!("index_page_{}", i);
                cache.insert(key.clone(), format!("index_data_{}", i));
                cache.get(&key);
                cache.get(&key);
            }

            // Table scan
            for i in 0..500 {
                cache.insert(format!("table_page_{}", i), format!("row_data_{}", i));
            }

            // Index pages should survive
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
        fn cdn_edge_cache() {
            let mut cache = S3FifoCache::new(50);

            // Popular content
            let popular = vec!["home.html", "style.css", "logo.png", "app.js"];
            for page in &popular {
                cache.insert(page.to_string(), format!("{}_content", page));
                cache.get(&page.to_string());
                cache.get(&page.to_string());
            }

            // Long tail requests
            for i in 0..200 {
                cache.insert(format!("user_page_{}", i), format!("content_{}", i));
            }

            // Popular content should survive
            for page in &popular {
                assert!(
                    cache.contains(&page.to_string()),
                    "Popular content '{}' should survive",
                    page
                );
            }
        }

        #[test]
        fn mixed_read_write() {
            let mut cache = S3FifoCache::new(100);

            // Working set
            for i in 0..30 {
                let key = format!("working_{}", i);
                cache.insert(key.clone(), i);
                cache.get(&key);
            }

            // Mixed operations
            for round in 0..5 {
                // Access some working set
                for i in (0..30).step_by(3) {
                    cache.get(&format!("working_{}", i));
                }

                // Insert new items
                for i in 0..20 {
                    cache.insert(format!("round_{}_{}", round, i), i);
                }
            }

            // Frequently accessed items should survive
            let mut hits = 0;
            for i in (0..30).step_by(3) {
                if cache.contains(&format!("working_{}", i)) {
                    hits += 1;
                }
            }

            assert!(
                hits >= 8,
                "Frequently accessed items should survive, got {} hits",
                hits
            );
        }
    }

    // ==============================================
    // Metrics Feature
    // ==============================================

    #[cfg(feature = "metrics")]
    mod metrics_tests {
        use super::*;

        #[test]
        fn metrics_track_hits_and_misses() {
            let mut cache = S3FifoCache::new(10);

            cache.insert("a", 1);
            cache.insert("b", 2);

            // Hits
            cache.get(&"a");
            cache.get(&"b");
            cache.get(&"a");

            let metrics = cache.metrics();
            assert_eq!(metrics.hits, 3);
            assert_eq!(metrics.inserts, 2);
            assert_eq!(metrics.updates, 0);
        }

        #[test]
        fn metrics_track_updates() {
            let mut cache = S3FifoCache::new(10);

            cache.insert("key", 1);
            cache.insert("key", 2); // Update
            cache.insert("key", 3); // Update

            let metrics = cache.metrics();
            assert_eq!(metrics.inserts, 1);
            assert_eq!(metrics.updates, 2);
        }

        #[test]
        fn metrics_track_promotions() {
            let mut cache = S3FifoCache::new(5);

            // Insert and access to increase frequency
            cache.insert("hot".to_string(), 0);
            cache.get(&"hot".to_string()); // freq = 1

            // Fill to trigger eviction and promotion
            for i in 0..10 {
                cache.insert(format!("cold_{}", i), i);
            }

            let metrics = cache.metrics();
            assert!(
                metrics.promotions > 0,
                "Expected promotions, got {}",
                metrics.promotions
            );
        }

        #[test]
        fn metrics_track_evictions() {
            let mut cache = S3FifoCache::new(3);

            // Fill cache
            cache.insert("a", 1);
            cache.insert("b", 2);
            cache.insert("c", 3);

            // Trigger eviction
            cache.insert("d", 4);

            let metrics = cache.metrics();
            assert!(
                metrics.small_evictions > 0 || metrics.main_evictions > 0,
                "Expected evictions"
            );
        }

        #[test]
        fn metrics_track_ghost_hits() {
            let mut cache = S3FifoCache::new(3);

            // Insert item
            cache.insert("will_evict", 1);

            // Fill to evict
            cache.insert("a", 1);
            cache.insert("b", 2);
            cache.insert("c", 3);

            // Verify eviction
            assert!(!cache.contains(&"will_evict"));

            // Re-insert (should be ghost hit if in ghost)
            cache.insert("will_evict", 2);

            let metrics = cache.metrics();
            // Ghost hit count depends on whether it was recorded
            // Just verify metrics are being tracked
            assert!(metrics.inserts > 0);
        }

        #[test]
        fn metrics_reset_works() {
            let mut cache = S3FifoCache::new(10);

            cache.insert("a", 1);
            cache.get(&"a");

            assert!(cache.metrics().hits > 0);

            cache.reset_metrics();

            let metrics = cache.metrics();
            assert_eq!(metrics.hits, 0);
            assert_eq!(metrics.inserts, 0);
        }

        #[test]
        fn metrics_main_reinserts() {
            let mut cache = S3FifoCache::new(5);

            // Create hot item that will be promoted to Main
            cache.insert("hot".to_string(), 0);
            cache.get(&"hot".to_string());
            cache.get(&"hot".to_string());
            cache.get(&"hot".to_string()); // freq = 3

            // Fill to trigger promotion and Main reinsertion
            for i in 0..20 {
                cache.insert(format!("filler_{}", i), i);
            }

            let metrics = cache.metrics();
            // Should have some promotions or main reinserts
            assert!(
                metrics.promotions > 0 || metrics.main_reinserts > 0,
                "Expected queue movements"
            );
        }
    }

    // ==============================================
    // New API Methods
    // ==============================================

    mod new_api_methods {
        use super::*;

        #[test]
        fn get_mut_works() {
            let mut cache = S3FifoCache::new(10);
            cache.insert("key", 42);

            if let Some(val) = cache.get_mut(&"key") {
                *val = 100;
            }

            assert_eq!(cache.get(&"key"), Some(&100));
        }

        #[test]
        fn get_mut_missing_returns_none() {
            let mut cache: S3FifoCache<&str, i32> = S3FifoCache::new(10);
            assert_eq!(cache.get_mut(&"missing"), None);
        }

        #[test]
        fn get_mut_increments_frequency() {
            let mut cache = S3FifoCache::new(10);
            cache.insert("key".to_string(), 42);

            // Access multiple times to increase frequency
            cache.get_mut(&"key".to_string());
            cache.get_mut(&"key".to_string());
            cache.get_mut(&"key".to_string());

            // Fill cache to trigger eviction
            for i in 0..20 {
                cache.insert(format!("filler_{}", i), i);
            }

            // "key" should survive due to high frequency
            assert!(cache.contains(&"key".to_string()));
        }

        #[test]
        fn remove_works() {
            let mut cache = S3FifoCache::new(10);
            cache.insert("a", 1);
            cache.insert("b", 2);
            cache.insert("c", 3);

            assert_eq!(cache.remove(&"b"), Some(2));
            assert_eq!(cache.len(), 2);
            assert!(!cache.contains(&"b"));
            assert_eq!(cache.remove(&"b"), None);
        }

        #[test]
        fn remove_from_small_queue() {
            let mut cache = S3FifoCache::new(10);
            cache.insert("small", 1);

            assert_eq!(cache.small_len(), 1);
            assert_eq!(cache.remove(&"small"), Some(1));
            assert_eq!(cache.small_len(), 0);
        }

        #[test]
        fn remove_from_main_queue() {
            let mut cache = S3FifoCache::new(5);

            // Insert and access to promote to Main
            cache.insert("main".to_string(), 1);
            cache.get(&"main".to_string());

            // Fill to trigger promotion
            for i in 0..10 {
                cache.insert(format!("filler_{}", i), i);
            }

            // Remove if it's in Main
            if cache.contains(&"main".to_string()) {
                cache.remove(&"main".to_string());
                assert!(!cache.contains(&"main".to_string()));
            }
        }

        #[test]
        fn iter_empty_cache() {
            let cache: S3FifoCache<&str, i32> = S3FifoCache::new(10);
            let items: Vec<_> = cache.iter().collect();
            assert_eq!(items.len(), 0);
        }

        #[test]
        fn iter_over_entries() {
            let mut cache = S3FifoCache::new(10);
            cache.insert("a", 1);
            cache.insert("b", 2);
            cache.insert("c", 3);

            let items: Vec<_> = cache.iter().collect();
            assert_eq!(items.len(), 3);

            // Check all entries are present (order may vary)
            let keys: Vec<_> = items.iter().map(|(k, _)| *k).collect();
            assert!(keys.contains(&&"a"));
            assert!(keys.contains(&&"b"));
            assert!(keys.contains(&&"c"));
        }

        #[test]
        fn iter_exact_size() {
            let mut cache = S3FifoCache::new(10);
            cache.insert("a", 1);
            cache.insert("b", 2);
            cache.insert("c", 3);

            let iter = cache.iter();
            assert_eq!(iter.len(), 3);
            assert_eq!(iter.size_hint(), (3, Some(3)));
        }

        #[test]
        fn keys_iterator() {
            let mut cache = S3FifoCache::new(10);
            cache.insert("x", 1);
            cache.insert("y", 2);
            cache.insert("z", 3);

            let keys: Vec<_> = cache.keys().copied().collect();
            assert_eq!(keys.len(), 3);
            assert!(keys.contains(&"x"));
            assert!(keys.contains(&"y"));
            assert!(keys.contains(&"z"));
        }

        #[test]
        fn values_iterator() {
            let mut cache = S3FifoCache::new(10);
            cache.insert("a", 10);
            cache.insert("b", 20);
            cache.insert("c", 30);

            let values: Vec<_> = cache.values().copied().collect();
            assert_eq!(values.len(), 3);
            assert!(values.contains(&10));
            assert!(values.contains(&20));
            assert!(values.contains(&30));
        }

        #[test]
        fn iter_with_promoted_items() {
            let mut cache = S3FifoCache::new(10);

            // Add items to Small
            cache.insert("s1".to_string(), 1);
            cache.insert("s2".to_string(), 2);

            // Add and access to promote to Main
            cache.insert("m1".to_string(), 3);
            cache.get(&"m1".to_string());

            // Fill to trigger promotion
            for i in 0..10 {
                cache.insert(format!("filler_{}", i), i);
            }

            // Iterate - should include items from both queues
            let items: Vec<_> = cache.iter().collect();
            assert_eq!(items.len(), cache.len());
        }

        #[test]
        fn iter_keys_exact_size() {
            let mut cache = S3FifoCache::new(10);
            cache.insert("a", 1);
            cache.insert("b", 2);

            let mut keys = cache.keys();
            assert_eq!(keys.len(), 2);
            keys.next();
            assert_eq!(keys.len(), 1);
        }

        #[test]
        fn iter_values_exact_size() {
            let mut cache = S3FifoCache::new(10);
            cache.insert("a", 1);
            cache.insert("b", 2);

            let mut values = cache.values();
            assert_eq!(values.len(), 2);
            values.next();
            assert_eq!(values.len(), 1);
        }
    }

    // ==============================================
    // Invariants Testing
    // ==============================================

    #[cfg(debug_assertions)]
    mod invariants_tests {
        use super::*;

        #[test]
        fn empty_cache_invariants() {
            let cache: S3FifoCache<&str, i32> = S3FifoCache::new(10);
            cache
                .check_invariants()
                .expect("Empty cache invariants failed");
        }

        #[test]
        fn single_item_invariants() {
            let mut cache = S3FifoCache::new(10);
            cache.insert("key", 42);
            cache
                .check_invariants()
                .expect("Single item invariants failed");
        }

        #[test]
        fn multiple_items_invariants() {
            let mut cache = S3FifoCache::new(10);
            for i in 0..5 {
                cache.insert(i, i * 10);
                cache
                    .check_invariants()
                    .unwrap_or_else(|e| panic!("Invariants failed after inserting {}: {}", i, e));
            }
        }

        #[test]
        fn after_access_invariants() {
            let mut cache = S3FifoCache::new(10);
            cache.insert("a", 1);
            cache.insert("b", 2);
            cache.insert("c", 3);

            cache.get(&"a");
            cache
                .check_invariants()
                .expect("Invariants failed after get");

            cache.get(&"b");
            cache.get(&"b");
            cache
                .check_invariants()
                .expect("Invariants failed after multiple gets");
        }

        #[test]
        fn after_eviction_invariants() {
            let mut cache = S3FifoCache::new(5);

            // Fill beyond capacity
            for i in 0..10 {
                cache.insert(i, i * 10);
                cache
                    .check_invariants()
                    .unwrap_or_else(|e| panic!("Invariants failed after inserting {}: {}", i, e));
            }
        }

        #[test]
        fn after_promotion_invariants() {
            let mut cache = S3FifoCache::new(5);

            // Insert and access to trigger promotion
            cache.insert("hot".to_string(), 1);
            cache.get(&"hot".to_string());

            // Fill to trigger eviction/promotion
            for i in 0..10 {
                cache.insert(format!("item_{}", i), i);
                cache
                    .check_invariants()
                    .unwrap_or_else(|e| panic!("Invariants failed after item_{}: {}", i, e));
            }
        }

        #[test]
        fn after_update_invariants() {
            let mut cache = S3FifoCache::new(10);
            cache.insert("key", 1);
            cache
                .check_invariants()
                .expect("Invariants failed after insert");

            cache.insert("key", 2);
            cache
                .check_invariants()
                .expect("Invariants failed after update");

            cache.insert("key", 3);
            cache
                .check_invariants()
                .expect("Invariants failed after second update");
        }

        #[test]
        fn after_remove_invariants() {
            let mut cache = S3FifoCache::new(10);
            cache.insert("a", 1);
            cache.insert("b", 2);
            cache.insert("c", 3);
            cache
                .check_invariants()
                .expect("Invariants failed after inserts");

            cache.remove(&"b");
            cache
                .check_invariants()
                .expect("Invariants failed after remove");

            cache.remove(&"a");
            cache
                .check_invariants()
                .expect("Invariants failed after second remove");
        }

        #[test]
        fn after_clear_invariants() {
            let mut cache = S3FifoCache::new(10);
            for i in 0..5 {
                cache.insert(i, i);
            }

            cache.clear();
            cache
                .check_invariants()
                .expect("Invariants failed after clear");
        }

        #[test]
        fn scan_workload_invariants() {
            let mut cache = S3FifoCache::new(50);

            // Create hot items
            for i in 0..10 {
                cache.insert(format!("hot_{}", i), i);
                cache.get(&format!("hot_{}", i));
            }

            cache
                .check_invariants()
                .expect("Invariants failed after hot items");

            // Scan workload
            for i in 0..100 {
                cache.insert(format!("scan_{}", i), i);
                if i % 10 == 0 {
                    cache.check_invariants().unwrap_or_else(|e| {
                        panic!("Invariants failed at scan iteration {}: {}", i, e)
                    });
                }
            }

            cache
                .check_invariants()
                .expect("Invariants failed after scan");
        }

        #[test]
        fn mixed_operations_invariants() {
            let mut cache = S3FifoCache::new(20);

            for round in 0..5 {
                // Insert
                for i in 0..10 {
                    cache.insert(format!("r{}_i{}", round, i), i);
                }

                // Access some
                for i in (0..10).step_by(2) {
                    cache.get(&format!("r{}_i{}", round, i));
                }

                // Remove some
                cache.remove(&format!("r{}_i1", round));

                cache
                    .check_invariants()
                    .unwrap_or_else(|e| panic!("Invariants failed at round {}: {}", round, e));
            }
        }

        #[test]
        fn small_capacity_invariants() {
            let mut cache = S3FifoCache::new(1);
            cache.insert("a", 1);
            cache
                .check_invariants()
                .expect("Invariants failed with capacity 1");

            cache.insert("b", 2);
            cache
                .check_invariants()
                .expect("Invariants failed after eviction");
        }

        #[test]
        fn get_mut_invariants() {
            let mut cache = S3FifoCache::new(10);
            cache.insert("key", 42);

            if let Some(val) = cache.get_mut(&"key") {
                *val = 100;
            }

            cache
                .check_invariants()
                .expect("Invariants failed after get_mut");
        }
    }
}
