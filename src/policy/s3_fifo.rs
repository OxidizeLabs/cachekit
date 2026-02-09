//! S3-FIFO (Simple, Scalable, Scan-resistant FIFO) Cache
//!
//! A modern cache eviction policy that achieves scan resistance using three
//! FIFO queues without the complexity of LRU bookkeeping. Nodes are stored
//! in a contiguous [`SlotArena`] with `Option<SlotId>` index links,
//! eliminating per-entry heap allocation and all `unsafe` code.
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────────┐
//! │                     S3FifoCache<K, V> Layout                       │
//! │                                                                         │
//! │   map: HashMap<K, SlotId>        arena: SlotArena<Node<K,V>>            │
//! │   ┌──────────┬──────────┐        ┌─────┬──────────────────────────┐     │
//! │   │   Key    │  SlotId  │        │ Idx │  key, value, freq, links │     │
//! │   ├──────────┼──────────┤        ├─────┼──────────────────────────┤     │
//! │   │  "pg_1"  │   id(0)  │───────►│  0  │  pg_1, data, 0          │     │
//! │   │  "pg_2"  │   id(1)  │───────►│  1  │  pg_2, data, 1          │     │
//! │   │  "pg_3"  │   id(2)  │───────►│  2  │  pg_3, data, 0          │     │
//! │   └──────────┴──────────┘        └─────┴──────────────────────────┘     │
//! │                                                                         │
//! │   Queue Organisation (same as S3FifoCache, but index-linked)            │
//! │                                                                         │
//! │   SMALL QUEUE: small_head ──► ... ──► small_tail                        │
//! │   MAIN  QUEUE: main_head  ──► ... ──► main_tail                         │
//! │   GHOST LIST:  GhostList<K> (keys only)                                 │
//! └──────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Performance
//!
//! | Aspect             | Detail                                        |
//! |--------------------|-----------------------------------------------|
//! | Allocation         | Vec grow (amortised), slot reuse               |
//! | Memory layout      | Contiguous Vec                                 |
//! | Unsafe code        | None                                           |
//! | Concurrent get     | Read-lock + atomic frequency bump              |
//!
//! ## Thread Safety
//!
//! - [`S3FifoCache`]: Single-threaded; requires `&mut self` for mutations.
//! - [`ConcurrentS3FifoCache`]: Thread-safe wrapper using `RwLock`.
//!   Reads (`get`, `get_with`) hold a **read lock** and bump frequency via
//!   `AtomicU8`, so multiple readers proceed in parallel.
//!
//! ## Example Usage
//!
//! ```
//! use cachekit::policy::s3_fifo::S3FifoCache;
//! use cachekit::traits::CoreCache;
//!
//! let mut cache: S3FifoCache<String, String> = S3FifoCache::new(100);
//!
//! cache.insert("page1".to_string(), "content1".to_string());
//! cache.insert("page2".to_string(), "content2".to_string());
//!
//! assert_eq!(cache.get(&"page1".to_string()), Some(&"content1".to_string()));
//!
//! for i in 0..150 {
//!     cache.insert(format!("scan_{}", i), format!("data_{}", i));
//! }
//!
//! assert_eq!(cache.len(), 100);
//! ```

use std::fmt::Debug;
use std::hash::Hash;
#[cfg(feature = "concurrency")]
use std::sync::Arc;
#[cfg(all(feature = "concurrency", feature = "metrics"))]
use std::sync::atomic::AtomicU64;
use std::sync::atomic::{AtomicU8, Ordering};

#[cfg(feature = "concurrency")]
use parking_lot::RwLock;
use rustc_hash::FxHashMap;

use crate::ds::{GhostList, SlotArena, SlotId};
use crate::error::ConfigError;
#[cfg(feature = "concurrency")]
use crate::traits::ConcurrentCache;
/// Performance metrics for S3-FIFO cache operations.
#[cfg(feature = "metrics")]
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

#[cfg(feature = "metrics")]
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
use crate::traits::CoreCache;
use crate::traits::MutableCache;
use crate::traits::ReadOnlyCache;

/// Maximum frequency value (2 bits = 0-3).
const MAX_FREQ: u8 = 3;

/// Default ratio of capacity allocated to Small queue.
const DEFAULT_SMALL_RATIO: f64 = 0.1;

/// Default ratio of capacity for Ghost list.
const DEFAULT_GHOST_RATIO: f64 = 0.9;

/// Which queue a node belongs to.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum QueueKind {
    Small,
    Main,
}

/// Internal node storing key, value, and metadata.
///
/// Links use `Option<SlotId>` indices into the parent `SlotArena` rather
/// than raw pointers, eliminating all `unsafe` code.
///
/// `freq` is an [`AtomicU8`] so that the concurrent wrapper can bump
/// frequency under a **read lock** (`get_shared`), avoiding a write lock
/// on the hot read path.
struct Node<K, V> {
    prev: Option<SlotId>,
    next: Option<SlotId>,
    queue: QueueKind,
    freq: AtomicU8,
    key: K,
    value: V,
}

// ---------------------------------------------------------------------------
// Iterators
// ---------------------------------------------------------------------------

/// Which queue an iterator is currently traversing.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum IterQueue {
    Small,
    Main,
}

/// Iterator over cache entries.
pub struct Iter<'a, K, V> {
    arena: &'a SlotArena<Node<K, V>>,
    current: Option<SlotId>,
    queue: IterQueue,
    main_head: Option<SlotId>,
    remaining: usize,
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        while self.remaining > 0 {
            match self.current {
                Some(id) => {
                    let node = self.arena.get(id).expect("iterator: stale SlotId");
                    self.current = node.next;
                    self.remaining -= 1;
                    return Some((&node.key, &node.value));
                },
                None => {
                    if self.queue == IterQueue::Small {
                        self.queue = IterQueue::Main;
                        self.current = self.main_head;
                    } else {
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

impl<K, V> ExactSizeIterator for Iter<'_, K, V> {
    fn len(&self) -> usize {
        self.remaining
    }
}

impl<K, V> std::iter::FusedIterator for Iter<'_, K, V> {}

impl<K, V> Debug for Iter<'_, K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Iter")
            .field("remaining", &self.remaining)
            .finish()
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

impl<K, V> ExactSizeIterator for Keys<'_, K, V> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<K, V> std::iter::FusedIterator for Keys<'_, K, V> {}

impl<K, V> Debug for Keys<'_, K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Keys")
            .field("remaining", &self.inner.remaining)
            .finish()
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

impl<K, V> ExactSizeIterator for Values<'_, K, V> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<K, V> std::iter::FusedIterator for Values<'_, K, V> {}

impl<K, V> Debug for Values<'_, K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Values")
            .field("remaining", &self.inner.remaining)
            .finish()
    }
}

/// Consuming iterator over cache entries.
pub struct IntoIter<K, V> {
    arena: SlotArena<Node<K, V>>,
    current: Option<SlotId>,
    main_head: Option<SlotId>,
    remaining: usize,
}

impl<K, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        let id = self.current.or_else(|| {
            // Finished Small, switch to Main once
            let mh = self.main_head.take();
            self.current = mh;
            mh
        })?;

        let node = self.arena.remove(id).expect("into_iter: stale SlotId");
        self.current = node.next;
        self.remaining -= 1;
        Some((node.key, node.value))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<K, V> ExactSizeIterator for IntoIter<K, V> {
    fn len(&self) -> usize {
        self.remaining
    }
}

impl<K, V> std::iter::FusedIterator for IntoIter<K, V> {}

impl<K, V> Debug for IntoIter<K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IntoIter")
            .field("remaining", &self.remaining)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// S3FifoCache
// ---------------------------------------------------------------------------

/// S3-FIFO cache with arena-backed contiguous storage.
///
/// Uses three FIFO queues (Small, Main, Ghost) with frequency counters.
/// Nodes live in a contiguous [`SlotArena`] with index-based links,
/// eliminating per-entry heap allocation and all `unsafe` code.
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
/// cache.insert("hot_key".to_string(), "data".to_string());
/// cache.get(&"hot_key".to_string());
///
/// for i in 0..200 {
///     cache.insert(format!("scan_{}", i), "x".to_string());
/// }
///
/// let _ = cache.contains(&"hot_key".to_string());
/// ```
pub struct S3FifoCache<K, V> {
    /// Node storage — contiguous Vec with slot reuse.
    arena: SlotArena<Node<K, V>>,

    /// Key -> SlotId mapping.
    map: FxHashMap<K, SlotId>,

    /// Small queue (FIFO): head=newest, tail=oldest.
    small_head: Option<SlotId>,
    small_tail: Option<SlotId>,
    small_len: usize,

    /// Maximum entries in the Small queue.
    small_cap: usize,

    /// Main queue (FIFO): head=newest, tail=oldest.
    main_head: Option<SlotId>,
    main_tail: Option<SlotId>,
    main_len: usize,

    /// Ghost list for tracking evicted keys.
    ghost: GhostList<K>,

    /// Total cache capacity.
    capacity: usize,

    /// Performance metrics (gated behind feature flag).
    #[cfg(feature = "metrics")]
    metrics: S3FifoMetrics,
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
    /// # Panics
    ///
    /// Panics if `capacity` is zero.
    pub fn new(capacity: usize) -> Self {
        Self::with_ratios(capacity, DEFAULT_SMALL_RATIO, DEFAULT_GHOST_RATIO)
    }

    /// Creates a new cache with custom queue ratios.
    ///
    /// # Panics
    ///
    /// Panics if parameters are invalid. See [`try_with_ratios`](Self::try_with_ratios).
    pub fn with_ratios(capacity: usize, small_ratio: f64, ghost_ratio: f64) -> Self {
        match Self::try_with_ratios(capacity, small_ratio, ghost_ratio) {
            Ok(cache) => cache,
            Err(e) => panic!("{}", e),
        }
    }

    /// Creates a new cache with custom ratios, returning an error on invalid parameters.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] if capacity is zero, `small_ratio` is not in
    /// `[0.0, 1.0]`, or `ghost_ratio` is negative/non-finite.
    pub fn try_with_ratios(
        capacity: usize,
        small_ratio: f64,
        ghost_ratio: f64,
    ) -> Result<Self, ConfigError> {
        if capacity == 0 {
            return Err(ConfigError::new("cache capacity must be greater than zero"));
        }
        if !small_ratio.is_finite() || !(0.0..=1.0).contains(&small_ratio) {
            return Err(ConfigError::new(format!(
                "small_ratio must be in [0.0, 1.0], got {}",
                small_ratio
            )));
        }
        if !ghost_ratio.is_finite() || ghost_ratio < 0.0 {
            return Err(ConfigError::new(format!(
                "ghost_ratio must be finite and non-negative, got {}",
                ghost_ratio
            )));
        }

        let small_cap = (capacity as f64 * small_ratio).round() as usize;
        let ghost_cap = (capacity as f64 * ghost_ratio).round() as usize;

        Ok(Self {
            arena: SlotArena::with_capacity(capacity),
            map: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
            small_head: None,
            small_tail: None,
            small_len: 0,
            small_cap,
            main_head: None,
            main_tail: None,
            main_len: 0,
            ghost: GhostList::new(ghost_cap),
            capacity,
            #[cfg(feature = "metrics")]
            metrics: S3FifoMetrics::default(),
        })
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
    #[inline]
    pub fn peek(&self, key: &K) -> Option<&V> {
        let &id = self.map.get(key)?;
        let node = self.arena.get(id)?;
        Some(&node.value)
    }

    /// Returns the number of entries in the Small queue.
    #[inline]
    pub fn small_len(&self) -> usize {
        self.small_len
    }

    /// Returns the maximum capacity of the Small queue.
    #[inline]
    pub fn small_capacity(&self) -> usize {
        self.small_cap
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
    #[inline]
    pub fn get(&mut self, key: &K) -> Option<&V> {
        let &id = match self.map.get(key) {
            Some(id) => id,
            None => {
                #[cfg(feature = "metrics")]
                {
                    self.metrics.misses += 1;
                }
                return None;
            },
        };

        #[cfg(feature = "metrics")]
        {
            self.metrics.hits += 1;
        }

        let node = self.arena.get(id).expect("map/arena out of sync");
        let f = node.freq.load(Ordering::Relaxed);
        if f < MAX_FREQ {
            node.freq.store(f + 1, Ordering::Relaxed);
        }
        Some(&node.value)
    }

    /// Retrieves a mutable reference to a value by key, incrementing its frequency.
    #[inline]
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        let &id = match self.map.get(key) {
            Some(id) => id,
            None => {
                #[cfg(feature = "metrics")]
                {
                    self.metrics.misses += 1;
                }
                return None;
            },
        };

        #[cfg(feature = "metrics")]
        {
            self.metrics.hits += 1;
        }

        let node = self.arena.get_mut(id).expect("map/arena out of sync");
        let freq = node.freq.get_mut();
        if *freq < MAX_FREQ {
            *freq += 1;
        }
        Some(&mut node.value)
    }

    /// Retrieves a value by key using only a shared reference, bumping
    /// frequency via an atomic compare-and-swap.
    ///
    /// This enables the concurrent wrapper to serve reads under a **read lock**
    /// while still updating the frequency counter. Metrics are **not** updated
    /// through this path; the concurrent wrapper maintains its own atomic
    /// hit/miss counters.
    #[cfg(feature = "concurrency")]
    #[inline]
    pub(crate) fn get_shared(&self, key: &K) -> Option<&V> {
        let &id = self.map.get(key)?;
        let node = self.arena.get(id).expect("map/arena out of sync");
        // Atomic freq bump via CAS loop: no lost updates under contention.
        let _ = node
            .freq
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |f| {
                if f < MAX_FREQ { Some(f + 1) } else { None }
            });
        Some(&node.value)
    }

    /// Inserts or updates a key-value pair.
    ///
    /// - If key exists: updates value and increments frequency
    /// - If key is in Ghost: inserts to Main queue
    /// - Otherwise: inserts to Small queue
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        // Update existing key
        if let Some(&id) = self.map.get(&key) {
            #[cfg(feature = "metrics")]
            {
                self.metrics.updates += 1;
            }
            let node = self.arena.get_mut(id).expect("map/arena out of sync");
            let old = std::mem::replace(&mut node.value, value);
            let freq = node.freq.get_mut();
            if *freq < MAX_FREQ {
                *freq += 1;
            }
            return Some(old);
        }

        #[cfg(feature = "metrics")]
        {
            self.metrics.inserts += 1;
        }

        // Ghost-guided admission
        let insert_to_main = self.ghost.remove(&key);

        #[cfg(feature = "metrics")]
        if insert_to_main {
            self.metrics.ghost_hits += 1;
        }

        // Evict before inserting
        self.evict_if_needed();

        let queue = if insert_to_main {
            QueueKind::Main
        } else {
            QueueKind::Small
        };

        let node = Node {
            prev: None,
            next: None,
            queue,
            freq: AtomicU8::new(0),
            key: key.clone(),
            value,
        };
        let id = self.arena.insert(node);
        self.map.insert(key, id);

        if insert_to_main {
            self.attach_main_head(id);
        } else {
            self.attach_small_head(id);
        }

        None
    }

    /// Removes a key-value pair from the cache.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let id = self.map.remove(key)?;
        let node = self.arena.get(id).expect("map/arena out of sync");
        let queue = node.queue;

        match queue {
            QueueKind::Small => self.detach_small(id),
            QueueKind::Main => self.detach_main(id),
        }

        let node = self.arena.remove(id).expect("map/arena out of sync");
        Some(node.value)
    }

    /// Clears all entries from the cache.
    pub fn clear(&mut self) {
        self.arena.clear();
        self.map.clear();
        self.ghost.clear();
        self.small_head = None;
        self.small_tail = None;
        self.small_len = 0;
        self.main_head = None;
        self.main_tail = None;
        self.main_len = 0;
    }

    /// Returns an iterator over all key-value pairs.
    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter {
            arena: &self.arena,
            current: self.small_head,
            queue: IterQueue::Small,
            main_head: self.main_head,
            remaining: self.len(),
        }
    }

    /// Returns an iterator over keys.
    pub fn keys(&self) -> Keys<'_, K, V> {
        Keys { inner: self.iter() }
    }

    /// Returns an iterator over values.
    pub fn values(&self) -> Values<'_, K, V> {
        Values { inner: self.iter() }
    }

    // -----------------------------------------------------------------------
    // Queue helpers
    // -----------------------------------------------------------------------

    #[inline]
    fn attach_small_head(&mut self, id: SlotId) {
        {
            let node = self.arena.get_mut(id).unwrap();
            node.prev = None;
            node.next = self.small_head;
            node.queue = QueueKind::Small;
        }

        if let Some(old_head) = self.small_head {
            self.arena.get_mut(old_head).unwrap().prev = Some(id);
        } else {
            self.small_tail = Some(id);
        }

        self.small_head = Some(id);
        self.small_len += 1;
    }

    #[inline]
    fn attach_main_head(&mut self, id: SlotId) {
        {
            let node = self.arena.get_mut(id).unwrap();
            node.prev = None;
            node.next = self.main_head;
            node.queue = QueueKind::Main;
        }

        if let Some(old_head) = self.main_head {
            self.arena.get_mut(old_head).unwrap().prev = Some(id);
        } else {
            self.main_tail = Some(id);
        }

        self.main_head = Some(id);
        self.main_len += 1;
    }

    fn detach_small(&mut self, id: SlotId) {
        let node = self.arena.get(id).unwrap();
        let prev = node.prev;
        let next = node.next;

        match prev {
            Some(p) => self.arena.get_mut(p).unwrap().next = next,
            None => self.small_head = next,
        }

        match next {
            Some(n) => self.arena.get_mut(n).unwrap().prev = prev,
            None => self.small_tail = prev,
        }

        self.small_len -= 1;
    }

    fn detach_main(&mut self, id: SlotId) {
        let node = self.arena.get(id).unwrap();
        let prev = node.prev;
        let next = node.next;

        match prev {
            Some(p) => self.arena.get_mut(p).unwrap().next = next,
            None => self.main_head = next,
        }

        match next {
            Some(n) => self.arena.get_mut(n).unwrap().prev = prev,
            None => self.main_tail = prev,
        }

        self.main_len -= 1;
    }

    /// Pop the tail node from the Small queue. Returns `(SlotId, freq)`.
    fn pop_small_tail(&mut self) -> Option<(SlotId, u8)> {
        let tail_id = self.small_tail?;
        let node = self.arena.get(tail_id).unwrap();
        let freq = node.freq.load(Ordering::Relaxed);
        let prev = node.prev;

        self.small_tail = prev;
        match prev {
            Some(p) => self.arena.get_mut(p).unwrap().next = None,
            None => self.small_head = None,
        }
        self.small_len -= 1;

        Some((tail_id, freq))
    }

    /// Pop the tail node from the Main queue. Returns `(SlotId, freq)`.
    fn pop_main_tail(&mut self) -> Option<(SlotId, u8)> {
        let tail_id = self.main_tail?;
        let node = self.arena.get(tail_id).unwrap();
        let freq = node.freq.load(Ordering::Relaxed);
        let prev = node.prev;

        self.main_tail = prev;
        match prev {
            Some(p) => self.arena.get_mut(p).unwrap().next = None,
            None => self.main_head = None,
        }
        self.main_len -= 1;

        Some((tail_id, freq))
    }

    // -----------------------------------------------------------------------
    // Eviction
    // -----------------------------------------------------------------------

    /// Attempts to evict or reinsert from the given queue.
    /// Returns `true` if an action occurred.
    fn try_evict_from_queue(&mut self, queue: QueueKind) -> bool {
        let popped = match queue {
            QueueKind::Small => self.pop_small_tail(),
            QueueKind::Main => self.pop_main_tail(),
        };

        let (id, freq) = match popped {
            Some(pair) => pair,
            None => return false,
        };

        if freq > 0 {
            // Promote / reinsert
            match queue {
                QueueKind::Small => {
                    #[cfg(feature = "metrics")]
                    {
                        self.metrics.promotions += 1;
                    }
                    *self.arena.get_mut(id).unwrap().freq.get_mut() = 0;
                    self.attach_main_head(id);
                },
                QueueKind::Main => {
                    #[cfg(feature = "metrics")]
                    {
                        self.metrics.main_reinserts += 1;
                    }
                    *self.arena.get_mut(id).unwrap().freq.get_mut() = freq - 1;
                    self.attach_main_head(id);
                },
            }
        } else {
            // Evict: freq == 0
            match queue {
                QueueKind::Small => {
                    #[cfg(feature = "metrics")]
                    {
                        self.metrics.small_evictions += 1;
                    }
                    let node = self.arena.remove(id).unwrap();
                    self.map.remove(&node.key);
                    self.ghost.record(node.key);
                },
                QueueKind::Main => {
                    #[cfg(feature = "metrics")]
                    {
                        self.metrics.main_evictions += 1;
                    }
                    let node = self.arena.remove(id).unwrap();
                    self.map.remove(&node.key);
                    // Don't record Main evictions in Ghost
                },
            }
        }

        true
    }

    /// Evicts entries until there is room for a new entry.
    fn evict_if_needed(&mut self) {
        while self.len() >= self.capacity {
            let acted = if self.small_len > self.small_cap {
                self.try_evict_from_queue(QueueKind::Small)
            } else if self.main_tail.is_some() {
                self.try_evict_from_queue(QueueKind::Main)
            } else {
                // Main is empty, fall back to Small
                self.try_evict_from_queue(QueueKind::Small)
            };

            if !acted {
                break;
            }
        }
    }

    /// Validates internal invariants (debug-only).
    #[cfg(debug_assertions)]
    pub fn check_invariants(&self) -> Result<(), crate::error::InvariantError>
    where
        K: Debug,
    {
        use crate::error::InvariantError;

        let total_len = self.small_len + self.main_len;
        if self.map.len() != total_len {
            return Err(InvariantError::new(format!(
                "Map size {} != small_len {} + main_len {} = {}",
                self.map.len(),
                self.small_len,
                self.main_len,
                total_len
            )));
        }

        // Small queue walk
        let mut count = 0;
        let mut current = self.small_head;
        let mut prev: Option<SlotId> = None;
        while let Some(id) = current {
            count += 1;
            let node = self.arena.get(id).ok_or_else(|| {
                InvariantError::new(format!("Small queue: stale SlotId {:?}", id))
            })?;
            if node.queue != QueueKind::Small {
                return Err(InvariantError::new(format!(
                    "Node {:?} in Small queue has queue={:?}",
                    node.key, node.queue
                )));
            }
            if node.prev != prev {
                return Err(InvariantError::new(format!(
                    "Small queue: node {:?} prev pointer inconsistent",
                    node.key
                )));
            }
            prev = Some(id);
            current = node.next;
        }
        if count != self.small_len {
            return Err(InvariantError::new(format!(
                "Small queue: counted {} but small_len = {}",
                count, self.small_len
            )));
        }

        // Main queue walk
        let mut count = 0;
        let mut current = self.main_head;
        let mut prev: Option<SlotId> = None;
        while let Some(id) = current {
            count += 1;
            let node = self
                .arena
                .get(id)
                .ok_or_else(|| InvariantError::new(format!("Main queue: stale SlotId {:?}", id)))?;
            if node.queue != QueueKind::Main {
                return Err(InvariantError::new(format!(
                    "Node {:?} in Main queue has queue={:?}",
                    node.key, node.queue
                )));
            }
            if node.prev != prev {
                return Err(InvariantError::new(format!(
                    "Main queue: node {:?} prev pointer inconsistent",
                    node.key
                )));
            }
            prev = Some(id);
            current = node.next;
        }
        if count != self.main_len {
            return Err(InvariantError::new(format!(
                "Main queue: counted {} but main_len = {}",
                count, self.main_len
            )));
        }

        if total_len > self.capacity {
            return Err(InvariantError::new(format!(
                "Total entries {} > capacity {}",
                total_len, self.capacity
            )));
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Std trait implementations
// ---------------------------------------------------------------------------

impl<K, V> FromIterator<(K, V)> for S3FifoCache<K, V>
where
    K: Clone + Eq + Hash,
{
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
    fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

impl<K, V> IntoIterator for S3FifoCache<K, V>
where
    K: Clone + Eq + Hash,
{
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;

    fn into_iter(mut self) -> Self::IntoIter {
        let small_head = self.small_head.take();
        let main_head = self.main_head.take();
        let remaining = self.len();

        // Take ownership of the arena; clear bookkeeping so Drop is a no-op.
        let arena = std::mem::take(&mut self.arena);
        self.small_tail = None;
        self.small_len = 0;
        self.main_tail = None;
        self.main_len = 0;
        self.map.clear();

        IntoIter {
            arena,
            current: small_head,
            main_head,
            remaining,
        }
    }
}

impl<'a, K, V> IntoIterator for &'a S3FifoCache<K, V>
where
    K: Clone + Eq + Hash,
{
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<K, V> Debug for S3FifoCache<K, V>
where
    K: Clone + Eq + Hash + Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("S3FifoCache")
            .field("capacity", &self.capacity)
            .field("len", &self.len())
            .field("small_len", &self.small_len)
            .field("small_cap", &self.small_cap)
            .field("main_len", &self.main_len)
            .field("ghost_len", &self.ghost.len())
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// Cache trait implementations
// ---------------------------------------------------------------------------

impl<K, V> ReadOnlyCache<K, V> for S3FifoCache<K, V>
where
    K: Clone + Eq + Hash,
{
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
}

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

    fn clear(&mut self) {
        S3FifoCache::clear(self);
    }
}

impl<K, V> MutableCache<K, V> for S3FifoCache<K, V>
where
    K: Clone + Eq + Hash,
{
    #[inline]
    fn remove(&mut self, key: &K) -> Option<V> {
        S3FifoCache::remove(self, key)
    }
}

// ---------------------------------------------------------------------------
// Concurrent wrapper
// ---------------------------------------------------------------------------

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
    ///
    /// # Panics
    ///
    /// Panics if the configured parameters are invalid (zero capacity,
    /// out-of-range ratios). For a non-panicking alternative, use
    /// [`try_build`](Self::try_build).
    pub fn build<K, V>(self) -> ConcurrentS3FifoCache<K, V>
    where
        K: Clone + Eq + Hash,
    {
        match self.try_build() {
            Ok(cache) => cache,
            Err(e) => panic!("{}", e),
        }
    }

    /// Builds the concurrent cache, returning an error on invalid parameters
    /// instead of panicking.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] if the configured capacity or ratios are invalid.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::s3_fifo::ConcurrentS3FifoCache;
    ///
    /// let cache = ConcurrentS3FifoCache::<String, i32>::builder(100)
    ///     .small_ratio(0.2)
    ///     .ghost_ratio(1.0)
    ///     .try_build::<String, i32>();
    /// assert!(cache.is_ok());
    /// ```
    pub fn try_build<K, V>(self) -> Result<ConcurrentS3FifoCache<K, V>, ConfigError>
    where
        K: Clone + Eq + Hash,
    {
        ConcurrentS3FifoCache::try_with_ratios(self.capacity, self.small_ratio, self.ghost_ratio)
    }
}

/// Thread-safe S3-FIFO cache wrapper using `RwLock`.
///
/// Provides concurrent access to an [`S3FifoCache`] with read-write
/// locking. Reads (`get`, `get_with`) acquire a **read lock** and bump
/// frequency atomically, so multiple readers proceed in parallel.
/// Mutations (`insert`, `remove`) acquire a **write lock**.
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
#[derive(Debug)]
pub struct ConcurrentS3FifoCache<K, V>
where
    K: Clone + Eq + Hash,
{
    inner: Arc<RwLock<S3FifoCache<K, V>>>,

    /// Hit counter for read-lock `get`/`get_with` path (not visible to inner cache).
    #[cfg(feature = "metrics")]
    read_hits: AtomicU64,

    /// Miss counter for read-lock `get`/`get_with` path (not visible to inner cache).
    #[cfg(feature = "metrics")]
    read_misses: AtomicU64,
}

#[cfg(feature = "concurrency")]
impl<K, V> Clone for ConcurrentS3FifoCache<K, V>
where
    K: Clone + Eq + Hash,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            #[cfg(feature = "metrics")]
            read_hits: AtomicU64::new(self.read_hits.load(Ordering::Relaxed)),
            #[cfg(feature = "metrics")]
            read_misses: AtomicU64::new(self.read_misses.load(Ordering::Relaxed)),
        }
    }
}

#[cfg(feature = "concurrency")]
impl<K, V> ConcurrentS3FifoCache<K, V>
where
    K: Clone + Eq + Hash,
{
    /// Creates a new concurrent S3-FIFO cache.
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(S3FifoCache::new(capacity))),
            #[cfg(feature = "metrics")]
            read_hits: AtomicU64::new(0),
            #[cfg(feature = "metrics")]
            read_misses: AtomicU64::new(0),
        }
    }

    /// Creates a new concurrent cache with custom ratios.
    ///
    /// # Panics
    ///
    /// Panics on invalid parameters. For a non-panicking alternative, use
    /// [`try_with_ratios`](Self::try_with_ratios) or the
    /// [`builder`](Self::builder) with [`try_build`](ConcurrentS3FifoCacheBuilder::try_build).
    pub fn with_ratios(capacity: usize, small_ratio: f64, ghost_ratio: f64) -> Self {
        match Self::try_with_ratios(capacity, small_ratio, ghost_ratio) {
            Ok(cache) => cache,
            Err(e) => panic!("{}", e),
        }
    }

    /// Creates a new concurrent cache with custom ratios, returning an
    /// error on invalid parameters instead of panicking.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] if any parameter is invalid.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::s3_fifo::ConcurrentS3FifoCache;
    ///
    /// let cache = ConcurrentS3FifoCache::<String, i32>::try_with_ratios(100, 0.2, 1.0);
    /// assert!(cache.is_ok());
    /// ```
    pub fn try_with_ratios(
        capacity: usize,
        small_ratio: f64,
        ghost_ratio: f64,
    ) -> Result<Self, ConfigError> {
        let inner = S3FifoCache::try_with_ratios(capacity, small_ratio, ghost_ratio)?;
        Ok(Self {
            inner: Arc::new(RwLock::new(inner)),
            #[cfg(feature = "metrics")]
            read_hits: AtomicU64::new(0),
            #[cfg(feature = "metrics")]
            read_misses: AtomicU64::new(0),
        })
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
    /// Uses a **read lock** so multiple `get` calls can proceed in parallel.
    /// The frequency counter is bumped atomically without exclusive access.
    ///
    /// This requires `V: Clone`. For non-cloneable values, use `get_with()`.
    pub fn get(&self, key: &K) -> Option<V>
    where
        V: Clone,
    {
        let guard = self.inner.read();
        let result = guard.get_shared(key);

        #[cfg(feature = "metrics")]
        {
            if result.is_some() {
                self.read_hits.fetch_add(1, Ordering::Relaxed);
            } else {
                self.read_misses.fetch_add(1, Ordering::Relaxed);
            }
        }

        result.cloned()
    }

    /// Gets a value by key and applies a function to it.
    ///
    /// Uses a **read lock** so multiple `get_with` calls can proceed in parallel.
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
        let guard = self.inner.read();
        let result = guard.get_shared(key);

        #[cfg(feature = "metrics")]
        {
            if result.is_some() {
                self.read_hits.fetch_add(1, Ordering::Relaxed);
            } else {
                self.read_misses.fetch_add(1, Ordering::Relaxed);
            }
        }

        result.map(f)
    }

    /// Peeks at a cloned value without updating frequency.
    ///
    /// Uses a **read lock**. Unlike `get`, this does not increment the
    /// frequency counter, making it safe for monitoring or debugging
    /// without influencing eviction order.
    pub fn peek(&self, key: &K) -> Option<V>
    where
        V: Clone,
    {
        self.inner.read().peek(key).cloned()
    }

    /// Peeks at a value without updating frequency or cloning.
    pub fn peek_with<F, R>(&self, key: &K, f: F) -> Option<R>
    where
        F: FnOnce(&V) -> R,
    {
        self.inner.read().peek(key).map(f)
    }

    /// Removes a key-value pair, returning the value if it existed.
    ///
    /// Uses a **write lock**.
    pub fn remove(&self, key: &K) -> Option<V> {
        self.inner.write().remove(key)
    }

    /// Removes multiple keys, returning the removed values in input order.
    ///
    /// Uses a **write lock** for the entire batch.
    pub fn remove_batch(&self, keys: &[K]) -> Vec<Option<V>> {
        let mut inner = self.inner.write();
        keys.iter().map(|k| inner.remove(k)).collect()
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

    /// Returns merged performance metrics (inner write-path + concurrent read-path).
    #[cfg(feature = "metrics")]
    pub fn metrics(&self) -> S3FifoMetrics {
        let mut m = self.inner.read().metrics().clone();
        m.hits += self.read_hits.load(Ordering::Relaxed);
        m.misses += self.read_misses.load(Ordering::Relaxed);
        m
    }

    /// Resets performance metrics to zero (both inner and concurrent counters).
    #[cfg(feature = "metrics")]
    pub fn reset_metrics(&self) {
        self.inner.write().reset_metrics();
        self.read_hits.store(0, Ordering::Relaxed);
        self.read_misses.store(0, Ordering::Relaxed);
    }
}

#[cfg(feature = "concurrency")]
impl<K, V> ConcurrentCache for ConcurrentS3FifoCache<K, V>
where
    K: Clone + Eq + Hash + Send + Sync,
    V: Send + Sync,
{
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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
            cache.get(&"a");
            cache.clear();
            assert!(cache.is_empty());
            assert_eq!(cache.len(), 0);
            assert!(!cache.contains(&"a"));
        }

        #[test]
        #[should_panic(expected = "cache capacity must be greater than zero")]
        fn zero_capacity_panics() {
            let _cache: S3FifoCache<&str, &str> = S3FifoCache::new(0);
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
            cache.insert("hot".to_string(), 0);
            cache.get(&"hot".to_string());

            for i in 1..10 {
                cache.insert(format!("cold_{}", i), i);
            }

            assert!(cache.contains(&"hot".to_string()));
        }

        #[test]
        fn unaccessed_items_evicted_first() {
            let mut cache: S3FifoCache<String, i32> = S3FifoCache::new(5);
            cache.insert("hot1".to_string(), 1);
            cache.get(&"hot1".to_string());
            cache.insert("hot2".to_string(), 2);
            cache.get(&"hot2".to_string());
            cache.insert("cold1".to_string(), 3);
            cache.insert("cold2".to_string(), 4);
            cache.insert("cold3".to_string(), 5);
            cache.insert("new".to_string(), 6);

            assert!(cache.contains(&"hot1".to_string()));
            assert!(cache.contains(&"hot2".to_string()));
            assert_eq!(cache.len(), 5);
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

            for i in 0..30 {
                let key = format!("working_{}", i);
                cache.insert(key.clone(), i);
                cache.get(&key);
            }

            for i in 0..200 {
                cache.insert(format!("scan_{}", i), i);
            }

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
    }

    // ==============================================
    // Eviction
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
        fn capacity_maintained() {
            let mut cache = S3FifoCache::new(100);
            for i in 0..1000 {
                cache.insert(i, i);
            }
            assert_eq!(cache.len(), 100);
        }

        #[test]
        fn single_capacity() {
            let mut cache = S3FifoCache::new(1);
            cache.insert("a", 1);
            assert!(cache.contains(&"a"));
            cache.insert("b", 2);
            assert!(!cache.contains(&"a"));
            assert!(cache.contains(&"b"));
        }
    }

    // ==============================================
    // Remove
    // ==============================================

    mod remove_tests {
        use super::*;

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
        fn get_mut_works() {
            let mut cache = S3FifoCache::new(10);
            cache.insert("key", 42);
            if let Some(val) = cache.get_mut(&"key") {
                *val = 100;
            }
            assert_eq!(cache.get(&"key"), Some(&100));
        }
    }

    // ==============================================
    // Iterators
    // ==============================================

    mod iterators {
        use super::*;

        #[test]
        fn iter_over_entries() {
            let mut cache = S3FifoCache::new(10);
            cache.insert("a", 1);
            cache.insert("b", 2);
            cache.insert("c", 3);
            let items: Vec<_> = cache.iter().collect();
            assert_eq!(items.len(), 3);
        }

        #[test]
        fn keys_iterator() {
            let mut cache = S3FifoCache::new(10);
            cache.insert("x", 1);
            cache.insert("y", 2);
            let keys: Vec<_> = cache.keys().copied().collect();
            assert_eq!(keys.len(), 2);
        }

        #[test]
        fn values_iterator() {
            let mut cache = S3FifoCache::new(10);
            cache.insert("a", 10);
            cache.insert("b", 20);
            let values: Vec<_> = cache.values().copied().collect();
            assert_eq!(values.len(), 2);
        }

        #[test]
        fn into_iter_yields_all() {
            let mut cache = S3FifoCache::new(10);
            cache.insert("a", 1);
            cache.insert("b", 2);
            cache.insert("c", 3);
            let mut items: Vec<_> = cache.into_iter().collect();
            items.sort_by_key(|(k, _)| *k);
            assert_eq!(items, vec![("a", 1), ("b", 2), ("c", 3)]);
        }

        #[test]
        fn from_iterator() {
            let cache: S3FifoCache<&str, i32> =
                vec![("a", 1), ("b", 2), ("c", 3)].into_iter().collect();
            assert_eq!(cache.len(), 3);
        }

        #[test]
        fn extend_adds_entries() {
            let mut cache = S3FifoCache::new(20);
            cache.insert("a", 1);
            cache.extend(vec![("b", 2), ("c", 3)]);
            assert_eq!(cache.len(), 3);
        }

        #[test]
        fn ref_for_loop() {
            let mut cache = S3FifoCache::new(10);
            cache.insert("a", 1);
            cache.insert("b", 2);
            let mut count = 0;
            for _ in &cache {
                count += 1;
            }
            assert_eq!(count, 2);
            assert_eq!(cache.len(), 2);
        }
    }

    // ==============================================
    // Invariants
    // ==============================================

    #[cfg(debug_assertions)]
    mod invariants {
        use super::*;

        #[test]
        fn after_operations() {
            let mut cache = S3FifoCache::new(20);

            for i in 0..10 {
                cache.insert(i, i * 10);
                cache.check_invariants().unwrap();
            }

            cache.get(&3);
            cache.get(&5);
            cache.check_invariants().unwrap();

            // Trigger eviction
            for i in 10..30 {
                cache.insert(i, i);
                cache.check_invariants().unwrap();
            }

            cache.remove(&15);
            cache.check_invariants().unwrap();

            cache.clear();
            cache.check_invariants().unwrap();
        }

        #[test]
        fn single_capacity_invariants() {
            let mut cache = S3FifoCache::new(1);
            cache.insert("a", 1);
            cache.check_invariants().unwrap();
            cache.insert("b", 2);
            cache.check_invariants().unwrap();
        }
    }

    // ==============================================
    // Leak Detection
    // ==============================================

    mod leak_detection {
        use super::*;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

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
        fn no_leak_on_eviction() {
            let counter = Arc::new(AtomicUsize::new(0));
            let mut cache = S3FifoCache::new(3);

            for i in 0..3 {
                cache.insert(i, LifeCycleTracker::new(i, counter.clone()));
            }
            assert_eq!(counter.load(Ordering::SeqCst), 3);

            cache.insert(99, LifeCycleTracker::new(99, counter.clone()));
            assert_eq!(counter.load(Ordering::SeqCst), 3);
            assert_eq!(cache.len(), 3);
        }

        #[test]
        fn no_leak_on_clear() {
            let counter = Arc::new(AtomicUsize::new(0));
            let mut cache = S3FifoCache::new(10);

            for i in 0..5 {
                cache.insert(i, LifeCycleTracker::new(i, counter.clone()));
            }
            assert_eq!(counter.load(Ordering::SeqCst), 5);
            cache.clear();
            assert_eq!(counter.load(Ordering::SeqCst), 0);
        }

        #[test]
        fn no_leak_on_drop() {
            let counter = Arc::new(AtomicUsize::new(0));
            {
                let mut cache = S3FifoCache::new(10);
                for i in 0..5 {
                    cache.insert(i, LifeCycleTracker::new(i, counter.clone()));
                }
                assert_eq!(counter.load(Ordering::SeqCst), 5);
            }
            assert_eq!(counter.load(Ordering::SeqCst), 0);
        }

        #[test]
        fn no_leak_on_remove() {
            let counter = Arc::new(AtomicUsize::new(0));
            let mut cache = S3FifoCache::new(10);

            cache.insert(1, LifeCycleTracker::new(1, counter.clone()));
            cache.insert(2, LifeCycleTracker::new(2, counter.clone()));
            assert_eq!(counter.load(Ordering::SeqCst), 2);

            drop(cache.remove(&1));
            assert_eq!(counter.load(Ordering::SeqCst), 1);
        }

        #[test]
        fn no_leak_on_update() {
            let counter = Arc::new(AtomicUsize::new(0));
            let mut cache = S3FifoCache::new(10);

            cache.insert(1, LifeCycleTracker::new(1, counter.clone()));
            assert_eq!(counter.load(Ordering::SeqCst), 1);

            let old = cache.insert(1, LifeCycleTracker::new(1, counter.clone()));
            drop(old);
            assert_eq!(counter.load(Ordering::SeqCst), 1);
        }

        #[test]
        fn no_leak_on_into_iter() {
            let counter = Arc::new(AtomicUsize::new(0));
            let mut cache = S3FifoCache::new(10);

            for i in 0..5 {
                cache.insert(i, LifeCycleTracker::new(i, counter.clone()));
            }
            assert_eq!(counter.load(Ordering::SeqCst), 5);

            let items: Vec<_> = cache.into_iter().collect();
            assert_eq!(items.len(), 5);
            assert_eq!(counter.load(Ordering::SeqCst), 5);
            drop(items);
            assert_eq!(counter.load(Ordering::SeqCst), 0);
        }

        #[test]
        fn no_leak_on_heavy_churn() {
            let counter = Arc::new(AtomicUsize::new(0));
            let mut cache = S3FifoCache::new(5);

            for i in 0..100 {
                cache.insert(i, LifeCycleTracker::new(i, counter.clone()));
            }

            let alive = counter.load(Ordering::SeqCst);
            assert_eq!(alive, cache.len());
            assert!(alive <= 5);

            drop(cache);
            assert_eq!(counter.load(Ordering::SeqCst), 0);
        }
    }

    // ==============================================
    // Trait Implementations
    // ==============================================

    mod trait_impls {
        use super::*;
        use crate::traits::{CoreCache, MutableCache};

        #[test]
        fn implements_core_cache() {
            fn assert_core_cache<T: CoreCache<K, V>, K, V>(_: &T) {}
            let cache: S3FifoCache<&str, i32> = S3FifoCache::new(10);
            assert_core_cache(&cache);
        }

        #[test]
        fn implements_mutable_cache() {
            fn assert_mutable_cache<T: MutableCache<K, V>, K, V>(_: &T) {}
            let cache: S3FifoCache<&str, i32> = S3FifoCache::new(10);
            assert_mutable_cache(&cache);
        }

        #[test]
        fn default_creates_cache() {
            let cache: S3FifoCache<String, i32> = S3FifoCache::default();
            assert!(cache.is_empty());
            assert_eq!(cache.capacity(), 128);
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

        #[test]
        fn concurrent_get_with() {
            let cache = ConcurrentS3FifoCache::new(10);
            cache.insert("key".to_string(), vec![1, 2, 3]);

            let len = cache.get_with(&"key".to_string(), |v| v.len());
            assert_eq!(len, Some(3));
        }

        #[test]
        fn concurrent_peek() {
            let cache = ConcurrentS3FifoCache::new(10);
            cache.insert("key".to_string(), 42);

            assert_eq!(cache.peek(&"key".to_string()), Some(42));
            assert_eq!(cache.peek(&"missing".to_string()), None);
        }

        #[test]
        fn concurrent_peek_with() {
            let cache = ConcurrentS3FifoCache::new(10);
            cache.insert("key".to_string(), vec![1, 2, 3]);

            let len = cache.peek_with(&"key".to_string(), |v| v.len());
            assert_eq!(len, Some(3));
        }

        #[test]
        fn concurrent_remove() {
            let cache = ConcurrentS3FifoCache::new(10);
            cache.insert("key".to_string(), 42);

            assert_eq!(cache.remove(&"key".to_string()), Some(42));
            assert_eq!(cache.remove(&"key".to_string()), None);
        }

        #[test]
        fn concurrent_remove_batch() {
            let cache = ConcurrentS3FifoCache::new(10);
            cache.insert("a".to_string(), 1);
            cache.insert("b".to_string(), 2);
            cache.insert("c".to_string(), 3);

            let removed = cache.remove_batch(&["a".to_string(), "z".to_string(), "c".to_string()]);
            assert_eq!(removed, vec![Some(1), None, Some(3)]);
            assert_eq!(cache.len(), 1);
        }

        #[test]
        fn concurrent_builder() {
            let cache: ConcurrentS3FifoCache<String, i32> =
                ConcurrentS3FifoCache::<String, i32>::builder(100)
                    .small_ratio(0.2)
                    .ghost_ratio(1.0)
                    .build();
            assert_eq!(cache.capacity(), 100);
        }

        #[test]
        fn concurrent_try_build() {
            let cache: Result<ConcurrentS3FifoCache<String, i32>, _> =
                ConcurrentS3FifoCache::<String, i32>::builder(100)
                    .small_ratio(0.2)
                    .ghost_ratio(1.0)
                    .try_build();
            assert!(cache.is_ok());
        }

        #[test]
        fn concurrent_clone_shares_state() {
            let cache = ConcurrentS3FifoCache::new(10);
            cache.insert("key".to_string(), 42);

            let cache2 = cache.clone();
            assert_eq!(cache2.get(&"key".to_string()), Some(42));

            cache.insert("new".to_string(), 99);
            assert_eq!(cache2.get(&"new".to_string()), Some(99));
        }

        #[test]
        fn concurrent_implements_trait() {
            fn assert_concurrent_cache<T: ConcurrentCache>(_: &T) {}
            let cache: ConcurrentS3FifoCache<String, i32> = ConcurrentS3FifoCache::new(10);
            assert_concurrent_cache(&cache);
        }
    }

    // ==============================================
    // Property Tests
    // ==============================================

    mod property_tests {
        use super::*;
        use proptest::prelude::*;

        #[derive(Debug, Clone)]
        enum Op {
            Insert(u32, u32),
            Get(u32),
            GetMut(u32),
            Remove(u32),
            Contains(u32),
        }

        fn op_strategy() -> impl Strategy<Value = Op> {
            prop_oneof![
                (0u32..50, any::<u32>()).prop_map(|(k, v)| Op::Insert(k, v)),
                (0u32..50).prop_map(Op::Get),
                (0u32..50).prop_map(Op::GetMut),
                (0u32..50).prop_map(Op::Remove),
                (0u32..50).prop_map(Op::Contains),
            ]
        }

        proptest! {
            #[cfg_attr(miri, ignore)]
            #[test]
            fn prop_invariants_always_hold(
                capacity in 1usize..30,
                ops in prop::collection::vec(op_strategy(), 0..100)
            ) {
                let mut cache: S3FifoCache<u32, u32> = S3FifoCache::new(capacity);
                for op in ops {
                    match op {
                        Op::Insert(k, v) => { cache.insert(k, v); },
                        Op::Get(k) => { cache.get(&k); },
                        Op::GetMut(k) => { cache.get_mut(&k); },
                        Op::Remove(k) => { cache.remove(&k); },
                        Op::Contains(k) => { cache.contains(&k); },
                    }
                    #[cfg(debug_assertions)]
                    cache.check_invariants().unwrap();
                }
            }

            #[cfg_attr(miri, ignore)]
            #[test]
            fn prop_len_never_exceeds_capacity(
                capacity in 1usize..30,
                keys in prop::collection::vec(0u32..100, 0..200)
            ) {
                let mut cache: S3FifoCache<u32, u32> = S3FifoCache::new(capacity);
                for k in keys {
                    cache.insert(k, k);
                    prop_assert!(cache.len() <= capacity);
                }
            }

            #[cfg_attr(miri, ignore)]
            #[test]
            fn prop_queue_lengths_sum_to_total(
                capacity in 1usize..30,
                ops in prop::collection::vec(op_strategy(), 0..100)
            ) {
                let mut cache: S3FifoCache<u32, u32> = S3FifoCache::new(capacity);
                for op in ops {
                    match op {
                        Op::Insert(k, v) => { cache.insert(k, v); },
                        Op::Get(k) => { cache.get(&k); },
                        Op::GetMut(k) => { cache.get_mut(&k); },
                        Op::Remove(k) => { cache.remove(&k); },
                        Op::Contains(k) => { cache.contains(&k); },
                    }
                    prop_assert_eq!(cache.small_len() + cache.main_len(), cache.len());
                }
            }
        }
    }
}
