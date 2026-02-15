//! Unified cache builder for all eviction policies.
//!
//! Provides a simple API to create caches with different eviction policies
//! while hiding internal implementation details (like `Arc<V>` wrapping).
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                              CacheBuilder                                   │
//! │                                                                             │
//! │   CacheBuilder::new(capacity)                                               │
//! │         │                                                                   │
//! │         ▼                                                                   │
//! │   .build::<K, V>(policy)                                                    │
//! │         │                                                                   │
//! │         ├─── CachePolicy::Fifo ────► FifoCache<K, V>                        │
//! │         ├─── CachePolicy::Lru ─────► LruCore<K, V>                          │
//! │         ├─── CachePolicy::FastLru ─► FastLru<K, V>                          │
//! │         ├─── CachePolicy::LruK ────► LrukCache<K, V>                        │
//! │         ├─── CachePolicy::Lfu ─────► LfuCache<K, V>                         │
//! │         ├─── CachePolicy::HeapLfu ─► HeapLfuCache<K, V>                     │
//! │         ├─── CachePolicy::TwoQ ────► TwoQCore<K, V>                         │
//! │         ├─── CachePolicy::S3Fifo ──► S3FifoCache<K, V>                      │
//! │         ├─── CachePolicy::Arc ─────► ARCCore<K, V>                          │
//! │         ├─── CachePolicy::Lifo ────► LifoCore<K, V>                         │
//! │         ├─── CachePolicy::Mfu ─────► MfuCore<K, V>                          │
//! │         ├─── CachePolicy::Mru ─────► MruCore<K, V>                          │
//! │         ├─── CachePolicy::Random ──► RandomCore<K, V>                       │
//! │         ├─── CachePolicy::Slru ────► SlruCore<K, V>                         │
//! │         ├─── CachePolicy::Clock ───► ClockCache<K, V>                       │
//! │         ├─── CachePolicy::ClockPro ► ClockProCache<K, V>                    │
//! │         └─── CachePolicy::Nru ─────► NruCache<K, V>                         │
//! │                                                                             │
//! │         ▼                                                                   │
//! │   Cache<K, V>  (unified wrapper)                                            │
//! │   ┌─────────────────────────────────────────────────────────────────────┐   │
//! │   │  .insert(key, value)  → Option<V>                                   │   │
//! │   │  .get(&key)           → Option<&V>                                  │   │
//! │   │  .contains(&key)      → bool                                        │   │
//! │   │  .len() / .is_empty() → usize / bool                                │   │
//! │   │  .capacity()          → usize                                       │   │
//! │   │  .clear()                                                           │   │
//! │   └─────────────────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Policy Comparison
//!
//! | Policy    | Best For                          | Eviction Basis        |
//! |-----------|-----------------------------------|-----------------------|
//! | FIFO      | Simple, predictable workloads     | Insertion order       |
//! | LRU       | Temporal locality                 | Recency               |
//! | FastLRU   | Maximum single-threaded speed     | Recency (no Arc)      |
//! | LRU-K     | Scan-resistant workloads          | K-th access time      |
//! | LFU       | Stable access patterns            | Frequency (O(1))      |
//! | HeapLFU   | Frequent evictions, large caches  | Frequency (O(log n))  |
//! | 2Q        | Mixed workloads                   | Two-queue promotion   |
//! | S3-FIFO   | CDN, scan-heavy workloads         | Three-queue FIFO      |
//! | ARC       | Adaptive recency/frequency        | Self-tuning           |
//! | LIFO      | Stack-like caching                | Reverse insertion     |
//! | MFU       | Inverse frequency patterns        | Highest frequency     |
//! | MRU       | Cyclic access patterns            | Most recent access    |
//! | Random    | Baseline comparisons              | Random selection      |
//! | SLRU      | Database buffer pools, scans      | Segmented LRU         |
//!
//! ## Example
//!
//! ```
//! use cachekit::builder::{CacheBuilder, CachePolicy};
//!
//! // Create an LRU cache
//! let mut cache = CacheBuilder::new(100).build::<u64, String>(CachePolicy::Lru);
//! cache.insert(1, "hello".to_string());
//! assert_eq!(cache.get(&1), Some(&"hello".to_string()));
//!
//! // Create an LRU-K cache for scan resistance
//! let mut cache = CacheBuilder::new(100).build::<u64, String>(CachePolicy::LruK { k: 2 });
//! cache.insert(1, "value".to_string());
//!
//! // Create a 2Q cache with 25% probation queue
//! let mut cache = CacheBuilder::new(100).build::<u64, String>(
//!     CachePolicy::TwoQ { probation_frac: 0.25 }
//! );
//! ```
//!
//! ## Type Constraints
//!
//! ```text
//! K: Copy + Eq + Hash + Ord
//!    │      │     │      │
//!    │      │     │      └── Required for HeapLFU (heap ordering)
//!    │      │     └───────── Required for HashMap indexing
//!    │      └─────────────── Required for key comparison
//!    └────────────────────── Required for efficient key handling
//!
//! V: Clone + Debug
//!    │       │
//!    │       └── Required for debug formatting
//!    └────────── Required for value extraction from Arc<V>
//! ```

use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;

#[cfg(feature = "policy-lfu")]
use crate::ds::frequency_buckets::DEFAULT_BUCKET_PREALLOC;
#[cfg(feature = "policy-arc")]
use crate::policy::arc::ARCCore;
#[cfg(feature = "policy-clock")]
use crate::policy::clock::ClockCache;
#[cfg(feature = "policy-clock-pro")]
use crate::policy::clock_pro::ClockProCache;
#[cfg(feature = "policy-fast-lru")]
use crate::policy::fast_lru::FastLru;
#[cfg(feature = "policy-fifo")]
use crate::policy::fifo::FifoCache;
#[cfg(feature = "policy-heap-lfu")]
use crate::policy::heap_lfu::HeapLfuCache;
#[cfg(feature = "policy-lfu")]
use crate::policy::lfu::LfuCache;
#[cfg(feature = "policy-lifo")]
use crate::policy::lifo::LifoCore;
#[cfg(feature = "policy-lru")]
use crate::policy::lru::LruCore;
#[cfg(feature = "policy-lru-k")]
use crate::policy::lru_k::LrukCache;
#[cfg(feature = "policy-mfu")]
use crate::policy::mfu::MfuCore;
#[cfg(feature = "policy-mru")]
use crate::policy::mru::MruCore;
#[cfg(feature = "policy-nru")]
use crate::policy::nru::NruCache;
#[cfg(feature = "policy-random")]
use crate::policy::random::RandomCore;
#[cfg(feature = "policy-s3-fifo")]
use crate::policy::s3_fifo::S3FifoCache;
#[cfg(feature = "policy-slru")]
use crate::policy::slru::SlruCore;
#[cfg(feature = "policy-two-q")]
use crate::policy::two_q::TwoQCore;
use crate::traits::{CoreCache, ReadOnlyCache};

/// Available cache eviction policies.
///
/// # Example
///
/// ```
/// use cachekit::builder::{CacheBuilder, CachePolicy};
///
/// // Simple FIFO for predictable eviction
/// let fifo = CacheBuilder::new(100).build::<u64, String>(CachePolicy::Fifo);
///
/// // LRU for temporal locality
/// let lru = CacheBuilder::new(100).build::<u64, String>(CachePolicy::Lru);
///
/// // LRU-K for scan resistance (K=2 is common)
/// let lru_k = CacheBuilder::new(100).build::<u64, String>(CachePolicy::LruK { k: 2 });
///
/// // LFU for stable access patterns (default bucket allocation)
/// let lfu = CacheBuilder::new(100).build::<u64, String>(CachePolicy::Lfu { bucket_hint: None });
///
/// // LFU with custom bucket pre-allocation for high-frequency workloads
/// let lfu = CacheBuilder::new(100).build::<u64, String>(CachePolicy::Lfu { bucket_hint: Some(64) });
///
/// // HeapLFU for large caches with frequent evictions
/// let heap_lfu = CacheBuilder::new(100).build::<u64, String>(CachePolicy::HeapLfu);
///
/// // 2Q for mixed workloads (25% probation queue)
/// let two_q = CacheBuilder::new(100).build::<u64, String>(
///     CachePolicy::TwoQ { probation_frac: 0.25 }
/// );
///
/// // S3-FIFO for scan-heavy workloads (10% small queue, 90% ghost list)
/// let s3_fifo = CacheBuilder::new(100).build::<u64, String>(
///     CachePolicy::S3Fifo { small_ratio: 0.1, ghost_ratio: 0.9 }
/// );
///
/// // LIFO for stack-like eviction
/// let lifo = CacheBuilder::new(100).build::<u64, String>(CachePolicy::Lifo);
///
/// // MFU for inverse frequency (evicts hot items)
/// let mfu = CacheBuilder::new(100).build::<u64, String>(CachePolicy::Mfu { bucket_hint: None });
///
/// // MRU for anti-recency patterns
/// let mru = CacheBuilder::new(100).build::<u64, String>(CachePolicy::Mru);
///
/// // Random for baseline comparisons
/// let random = CacheBuilder::new(100).build::<u64, String>(CachePolicy::Random);
///
/// // SLRU for scan resistance with two segments
/// let slru = CacheBuilder::new(100).build::<u64, String>(
///     CachePolicy::Slru { probationary_frac: 0.25 }
/// );
///
/// // ARC for adaptive recency/frequency balance
/// let arc = CacheBuilder::new(100).build::<u64, String>(CachePolicy::Arc);
///
/// // FastLru for maximum single-threaded performance
/// let fast_lru = CacheBuilder::new(100).build::<u64, String>(CachePolicy::FastLru);
/// ```
#[derive(Debug, Clone)]
pub enum CachePolicy {
    /// First In, First Out eviction.
    ///
    /// Evicts the oldest inserted item. Simple and predictable.
    /// Good for: streaming data, simple caching needs.
    #[cfg(feature = "policy-fifo")]
    Fifo,

    /// Least Recently Used eviction.
    ///
    /// Evicts the item that hasn't been accessed for the longest time.
    /// Good for: temporal locality, general-purpose caching.
    #[cfg(feature = "policy-lru")]
    Lru,

    /// Fast LRU eviction (optimized for single-threaded performance).
    ///
    /// Like LRU but stores values directly without Arc wrapping,
    /// using FxHash for faster operations (~7-10x faster than standard LRU).
    /// Good for: maximum single-threaded performance, values don't need to outlive eviction.
    #[cfg(feature = "policy-fast-lru")]
    FastLru,

    /// LRU-K policy with configurable K value.
    ///
    /// Tracks the K-th most recent access time for eviction decisions.
    /// Provides scan resistance (one-time accesses don't pollute cache).
    ///
    /// - `k: usize` - Number of accesses to track (K=2 is common)
    ///
    /// Good for: database buffer pools, scan-heavy workloads.
    #[cfg(feature = "policy-lru-k")]
    LruK { k: usize },

    /// Least Frequently Used eviction (bucket-based, O(1)).
    ///
    /// Evicts the item with the lowest access count.
    /// Uses frequency buckets for O(1) operations.
    ///
    /// - `bucket_hint: Option<usize>` - Pre-allocated frequency buckets (default: 32)
    ///
    /// Good for: stable access patterns, reference data.
    #[cfg(feature = "policy-lfu")]
    Lfu {
        /// Pre-allocated frequency buckets. Most items cluster at low frequencies,
        /// so the default (32) covers typical workloads. Increase for long-running
        /// caches with varied access patterns.
        bucket_hint: Option<usize>,
    },

    /// Least Frequently Used eviction (heap-based, O(log n)).
    ///
    /// Like LFU but uses a min-heap for eviction.
    /// Better for large caches with frequent evictions.
    ///
    /// Good for: high-throughput systems, large caches.
    #[cfg(feature = "policy-heap-lfu")]
    HeapLfu,

    /// Two-Queue policy with configurable probation fraction.
    ///
    /// Uses two queues: probation (for new items) and protected (for promoted items).
    /// Items are promoted after a second access.
    ///
    /// - `probation_frac: f64` - Fraction of capacity for probation queue (0.0-1.0)
    ///
    /// Good for: mixed workloads, scan resistance.
    #[cfg(feature = "policy-two-q")]
    TwoQ { probation_frac: f64 },

    /// S3-FIFO (Simple, Scalable, Scan-resistant FIFO) policy.
    ///
    /// Uses three FIFO queues: Small (for new items), Main (for promoted items),
    /// and Ghost (for tracking evicted keys). Provides excellent scan resistance
    /// with O(1) operations and minimal overhead.
    ///
    /// - `small_ratio: f64` - Fraction of capacity for Small queue (default 0.1)
    /// - `ghost_ratio: f64` - Fraction of capacity for Ghost list (default 0.9)
    ///
    /// Good for: CDN caches, scan-heavy workloads, database buffer pools.
    #[cfg(feature = "policy-s3-fifo")]
    S3Fifo {
        /// Fraction of capacity for the Small queue (filters one-hit wonders).
        small_ratio: f64,
        /// Fraction of capacity for the Ghost list (tracks evicted keys).
        ghost_ratio: f64,
    },

    /// Adaptive Replacement Cache eviction.
    ///
    /// Automatically adapts between recency (LRU-like) and frequency (LFU-like)
    /// preferences by maintaining four lists (T1, T2, B1, B2) and a dynamic
    /// target parameter. Provides excellent performance across diverse workloads.
    ///
    /// Good for: unknown or changing workloads, self-tuning caches.
    #[cfg(feature = "policy-arc")]
    Arc,

    /// Last In, First Out eviction.
    ///
    /// Evicts the most recently inserted item (stack-like behavior).
    /// Good for: Undo buffers, temporary scratch space.
    #[cfg(feature = "policy-lifo")]
    Lifo,

    /// Most Frequently Used eviction (bucket-based, O(1)).
    ///
    /// Evicts the item with the highest access count.
    /// Inverse of LFU - useful for specific niche workloads.
    ///
    /// - `bucket_hint: Option<usize>` - Pre-allocated frequency buckets (default: 32)
    ///
    /// Good for: Niche cases where most frequent = least needed next.
    #[cfg(feature = "policy-mfu")]
    Mfu {
        /// Pre-allocated frequency buckets for high-frequency items.
        bucket_hint: Option<usize>,
    },

    /// Most Recently Used eviction.
    ///
    /// Evicts the most recently accessed item (opposite of LRU).
    /// Good for: Cyclic access patterns, sequential scans.
    #[cfg(feature = "policy-mru")]
    Mru,

    /// Random eviction.
    ///
    /// Evicts a uniformly random item when capacity is reached.
    /// Good for: Baseline comparisons, truly random workloads.
    #[cfg(feature = "policy-random")]
    Random,

    /// Segmented LRU with probationary and protected segments.
    ///
    /// Uses two LRU queues: probationary (for new items) and protected (for promoted items).
    /// Items are promoted on re-access. Provides excellent scan resistance.
    ///
    /// - `probationary_frac: f64` - Fraction of capacity for probationary queue (0.0-1.0)
    ///
    /// Good for: Database buffer pools, scan-resistant workloads.
    #[cfg(feature = "policy-slru")]
    Slru {
        /// Fraction of capacity for the probationary segment.
        probationary_frac: f64,
    },

    /// Clock (Second-Chance) eviction.
    ///
    /// Approximates LRU using reference bits and a clock hand.
    /// Lower overhead than full LRU (no list manipulation on access).
    ///
    /// Good for: Low-latency caching, LRU approximation with lower overhead.
    #[cfg(feature = "policy-clock")]
    Clock,

    /// Clock-PRO eviction.
    ///
    /// Scan-resistant Clock variant with adaptive promotion.
    /// Combines Clock mechanics with ghost history tracking.
    ///
    /// Good for: Scan-heavy workloads, adaptive caching needs.
    #[cfg(feature = "policy-clock-pro")]
    ClockPro,

    /// NRU (Not Recently Used) eviction.
    ///
    /// Simple reference bit tracking with O(n) worst-case eviction.
    /// Coarser granularity than Clock, simpler implementation.
    ///
    /// Good for: Small-to-medium caches, simple coarse recency tracking.
    #[cfg(feature = "policy-nru")]
    Nru,
}

/// Unified cache wrapper that provides a consistent API regardless of policy.
///
/// Wraps different cache implementations behind a single interface.
/// All policy-specific details (like `Arc<V>` wrapping) are handled internally.
///
/// # Type Parameters
///
/// - `K`: Key type, must be `Copy + Eq + Hash + Ord`
/// - `V`: Value type, must be `Clone + Debug`
///
/// # Example
///
/// ```
/// use cachekit::builder::{CacheBuilder, CachePolicy};
///
/// let mut cache = CacheBuilder::new(3).build::<u64, String>(CachePolicy::Lru);
///
/// // Insert items
/// cache.insert(1, "one".to_string());
/// cache.insert(2, "two".to_string());
/// cache.insert(3, "three".to_string());
///
/// // Check existence (doesn't update LRU order)
/// assert!(cache.contains(&1));
/// assert!(cache.contains(&2));
///
/// // Check size
/// assert_eq!(cache.len(), 3);
/// assert_eq!(cache.capacity(), 3);
///
/// // Access key 2 to make it MRU
/// cache.get(&2);
///
/// // Eviction on insert: key 1 is now LRU
/// cache.insert(4, "four".to_string());
/// assert!(!cache.contains(&1));  // LRU item evicted
/// assert!(cache.contains(&2));   // Was accessed, survived
///
/// // Clear
/// cache.clear();
/// assert!(cache.is_empty());
/// ```
pub struct Cache<K, V>
where
    K: Copy + Eq + Hash + Ord,
    V: Clone + Debug,
{
    inner: CacheInner<K, V>,
}

enum CacheInner<K, V>
where
    K: Copy + Eq + Hash + Ord,
    V: Clone + Debug,
{
    #[cfg(feature = "policy-fifo")]
    Fifo(FifoCache<K, V>),
    #[cfg(feature = "policy-lru")]
    Lru(LruCore<K, V>),
    #[cfg(feature = "policy-fast-lru")]
    FastLru(FastLru<K, V>),
    #[cfg(feature = "policy-lru-k")]
    LruK(LrukCache<K, V>),
    #[cfg(feature = "policy-lfu")]
    Lfu(LfuCache<K, V>),
    #[cfg(feature = "policy-heap-lfu")]
    HeapLfu(HeapLfuCache<K, V>),
    #[cfg(feature = "policy-two-q")]
    TwoQ(TwoQCore<K, V>),
    #[cfg(feature = "policy-s3-fifo")]
    S3Fifo(S3FifoCache<K, V>),
    #[cfg(feature = "policy-arc")]
    Arc(ARCCore<K, V>),
    #[cfg(feature = "policy-lifo")]
    Lifo(LifoCore<K, V>),
    #[cfg(feature = "policy-mfu")]
    Mfu(MfuCore<K, V>),
    #[cfg(feature = "policy-mru")]
    Mru(MruCore<K, V>),
    #[cfg(feature = "policy-random")]
    Random(RandomCore<K, V>),
    #[cfg(feature = "policy-slru")]
    Slru(SlruCore<K, V>),
    #[cfg(feature = "policy-clock")]
    Clock(ClockCache<K, V>),
    #[cfg(feature = "policy-clock-pro")]
    ClockPro(ClockProCache<K, V>),
    #[cfg(feature = "policy-nru")]
    Nru(NruCache<K, V>),
}

impl<K, V> Cache<K, V>
where
    K: Copy + Eq + Hash + Ord,
    V: Clone + Debug,
{
    /// Inserts a key-value pair, returning the previous value if the key existed.
    ///
    /// If the cache is at capacity, evicts an item according to the policy.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::builder::{CacheBuilder, CachePolicy};
    ///
    /// let mut cache = CacheBuilder::new(10).build::<u64, String>(CachePolicy::Lru);
    ///
    /// // New insertion returns None
    /// assert_eq!(cache.insert(1, "one".to_string()), None);
    ///
    /// // Update returns previous value
    /// assert_eq!(cache.insert(1, "ONE".to_string()), Some("one".to_string()));
    /// assert_eq!(cache.get(&1), Some(&"ONE".to_string()));
    /// ```
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        match &mut self.inner {
            #[cfg(feature = "policy-fifo")]
            CacheInner::Fifo(fifo) => CoreCache::insert(fifo, key, value),
            #[cfg(feature = "policy-lru")]
            CacheInner::Lru(lru) => {
                let arc_value = Arc::new(value);
                lru.insert(key, arc_value)
                    .map(|arc| Arc::try_unwrap(arc).unwrap_or_else(|arc| (*arc).clone()))
            },
            #[cfg(feature = "policy-fast-lru")]
            CacheInner::FastLru(fast_lru) => fast_lru.insert(key, value),
            #[cfg(feature = "policy-lru-k")]
            CacheInner::LruK(lruk) => CoreCache::insert(lruk, key, value),
            #[cfg(feature = "policy-lfu")]
            CacheInner::Lfu(lfu) => {
                let arc_value = Arc::new(value);
                lfu.insert(key, arc_value)
                    .map(|arc| Arc::try_unwrap(arc).unwrap_or_else(|arc| (*arc).clone()))
            },
            #[cfg(feature = "policy-heap-lfu")]
            CacheInner::HeapLfu(heap_lfu) => {
                let arc_value = Arc::new(value);
                heap_lfu
                    .insert(key, arc_value)
                    .map(|arc| Arc::try_unwrap(arc).unwrap_or_else(|arc| (*arc).clone()))
            },
            #[cfg(feature = "policy-two-q")]
            CacheInner::TwoQ(twoq) => CoreCache::insert(twoq, key, value),
            #[cfg(feature = "policy-s3-fifo")]
            CacheInner::S3Fifo(s3fifo) => CoreCache::insert(s3fifo, key, value),
            #[cfg(feature = "policy-arc")]
            CacheInner::Arc(arc) => CoreCache::insert(arc, key, value),
            #[cfg(feature = "policy-lifo")]
            CacheInner::Lifo(lifo) => CoreCache::insert(lifo, key, value),
            #[cfg(feature = "policy-mfu")]
            CacheInner::Mfu(mfu) => CoreCache::insert(mfu, key, value),
            #[cfg(feature = "policy-mru")]
            CacheInner::Mru(mru) => CoreCache::insert(mru, key, value),
            #[cfg(feature = "policy-random")]
            CacheInner::Random(random) => CoreCache::insert(random, key, value),
            #[cfg(feature = "policy-slru")]
            CacheInner::Slru(slru) => CoreCache::insert(slru, key, value),
            #[cfg(feature = "policy-clock")]
            CacheInner::Clock(clock) => CoreCache::insert(clock, key, value),
            #[cfg(feature = "policy-clock-pro")]
            CacheInner::ClockPro(clock_pro) => CoreCache::insert(clock_pro, key, value),
            #[cfg(feature = "policy-nru")]
            CacheInner::Nru(nru) => CoreCache::insert(nru, key, value),
        }
    }

    /// Gets a reference to a value by key.
    ///
    /// Updates access metadata (recency/frequency) according to the policy.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::builder::{CacheBuilder, CachePolicy};
    ///
    /// let mut cache = CacheBuilder::new(10).build::<u64, String>(CachePolicy::Lru);
    /// cache.insert(1, "value".to_string());
    ///
    /// assert_eq!(cache.get(&1), Some(&"value".to_string()));
    /// assert_eq!(cache.get(&99), None);  // Missing key
    /// ```
    pub fn get(&mut self, key: &K) -> Option<&V> {
        match &mut self.inner {
            #[cfg(feature = "policy-fifo")]
            CacheInner::Fifo(fifo) => fifo.get(key),
            #[cfg(feature = "policy-lru")]
            CacheInner::Lru(lru) => lru.get(key).map(|arc| arc.as_ref()),
            #[cfg(feature = "policy-fast-lru")]
            CacheInner::FastLru(fast_lru) => fast_lru.get(key),
            #[cfg(feature = "policy-lru-k")]
            CacheInner::LruK(lruk) => lruk.get(key),
            #[cfg(feature = "policy-lfu")]
            CacheInner::Lfu(lfu) => lfu.get(key).map(|arc| arc.as_ref()),
            #[cfg(feature = "policy-heap-lfu")]
            CacheInner::HeapLfu(heap_lfu) => heap_lfu.get(key).map(|arc| arc.as_ref()),
            #[cfg(feature = "policy-two-q")]
            CacheInner::TwoQ(twoq) => twoq.get(key),
            #[cfg(feature = "policy-s3-fifo")]
            CacheInner::S3Fifo(s3fifo) => s3fifo.get(key),
            #[cfg(feature = "policy-arc")]
            CacheInner::Arc(arc) => arc.get(key),
            #[cfg(feature = "policy-lifo")]
            CacheInner::Lifo(lifo) => lifo.get(key),
            #[cfg(feature = "policy-mfu")]
            CacheInner::Mfu(mfu) => mfu.get(key),
            #[cfg(feature = "policy-mru")]
            CacheInner::Mru(mru) => mru.get(key),
            #[cfg(feature = "policy-random")]
            CacheInner::Random(random) => random.get(key),
            #[cfg(feature = "policy-slru")]
            CacheInner::Slru(slru) => slru.get(key),
            #[cfg(feature = "policy-clock")]
            CacheInner::Clock(clock) => clock.get(key),
            #[cfg(feature = "policy-clock-pro")]
            CacheInner::ClockPro(clock_pro) => clock_pro.get(key),
            #[cfg(feature = "policy-nru")]
            CacheInner::Nru(nru) => nru.get(key),
        }
    }

    /// Checks if a key exists in the cache.
    ///
    /// Does not update access metadata.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::builder::{CacheBuilder, CachePolicy};
    ///
    /// let mut cache = CacheBuilder::new(10).build::<u64, String>(CachePolicy::Lru);
    /// cache.insert(1, "value".to_string());
    ///
    /// assert!(cache.contains(&1));
    /// assert!(!cache.contains(&99));
    /// ```
    pub fn contains(&self, key: &K) -> bool {
        match &self.inner {
            #[cfg(feature = "policy-fifo")]
            CacheInner::Fifo(fifo) => fifo.contains(key),
            #[cfg(feature = "policy-lru")]
            CacheInner::Lru(lru) => lru.contains(key),
            #[cfg(feature = "policy-fast-lru")]
            CacheInner::FastLru(fast_lru) => fast_lru.contains(key),
            #[cfg(feature = "policy-lru-k")]
            CacheInner::LruK(lruk) => lruk.contains(key),
            #[cfg(feature = "policy-lfu")]
            CacheInner::Lfu(lfu) => lfu.contains(key),
            #[cfg(feature = "policy-heap-lfu")]
            CacheInner::HeapLfu(heap_lfu) => heap_lfu.contains(key),
            #[cfg(feature = "policy-two-q")]
            CacheInner::TwoQ(twoq) => twoq.contains(key),
            #[cfg(feature = "policy-s3-fifo")]
            CacheInner::S3Fifo(s3fifo) => s3fifo.contains(key),
            #[cfg(feature = "policy-arc")]
            CacheInner::Arc(arc) => arc.contains(key),
            #[cfg(feature = "policy-lifo")]
            CacheInner::Lifo(lifo) => lifo.contains(key),
            #[cfg(feature = "policy-mfu")]
            CacheInner::Mfu(mfu) => mfu.contains(key),
            #[cfg(feature = "policy-mru")]
            CacheInner::Mru(mru) => mru.contains(key),
            #[cfg(feature = "policy-random")]
            CacheInner::Random(random) => random.contains(key),
            #[cfg(feature = "policy-slru")]
            CacheInner::Slru(slru) => slru.contains(key),
            #[cfg(feature = "policy-clock")]
            CacheInner::Clock(clock) => clock.contains(key),
            #[cfg(feature = "policy-clock-pro")]
            CacheInner::ClockPro(clock_pro) => clock_pro.contains(key),
            #[cfg(feature = "policy-nru")]
            CacheInner::Nru(nru) => nru.contains(key),
        }
    }

    /// Returns the number of entries in the cache.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::builder::{CacheBuilder, CachePolicy};
    ///
    /// let mut cache = CacheBuilder::new(10).build::<u64, String>(CachePolicy::Lru);
    /// assert_eq!(cache.len(), 0);
    ///
    /// cache.insert(1, "one".to_string());
    /// cache.insert(2, "two".to_string());
    /// assert_eq!(cache.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        match &self.inner {
            #[cfg(feature = "policy-fifo")]
            CacheInner::Fifo(fifo) => <dyn CoreCache<K, V>>::len(fifo),
            #[cfg(feature = "policy-lru")]
            CacheInner::Lru(lru) => lru.len(),
            #[cfg(feature = "policy-fast-lru")]
            CacheInner::FastLru(fast_lru) => fast_lru.len(),
            #[cfg(feature = "policy-lru-k")]
            CacheInner::LruK(lruk) => <dyn CoreCache<K, V>>::len(lruk),
            #[cfg(feature = "policy-lfu")]
            CacheInner::Lfu(lfu) => lfu.len(),
            #[cfg(feature = "policy-heap-lfu")]
            CacheInner::HeapLfu(heap_lfu) => heap_lfu.len(),
            #[cfg(feature = "policy-two-q")]
            CacheInner::TwoQ(twoq) => twoq.len(),
            #[cfg(feature = "policy-s3-fifo")]
            CacheInner::S3Fifo(s3fifo) => s3fifo.len(),
            #[cfg(feature = "policy-arc")]
            CacheInner::Arc(arc) => arc.len(),
            #[cfg(feature = "policy-lifo")]
            CacheInner::Lifo(lifo) => <dyn CoreCache<K, V>>::len(lifo),
            #[cfg(feature = "policy-mfu")]
            CacheInner::Mfu(mfu) => mfu.len(),
            #[cfg(feature = "policy-mru")]
            CacheInner::Mru(mru) => mru.len(),
            #[cfg(feature = "policy-random")]
            CacheInner::Random(random) => <dyn CoreCache<K, V>>::len(random),
            #[cfg(feature = "policy-slru")]
            CacheInner::Slru(slru) => slru.len(),
            #[cfg(feature = "policy-clock")]
            CacheInner::Clock(clock) => <dyn CoreCache<K, V>>::len(clock),
            #[cfg(feature = "policy-clock-pro")]
            CacheInner::ClockPro(clock_pro) => <dyn CoreCache<K, V>>::len(clock_pro),
            #[cfg(feature = "policy-nru")]
            CacheInner::Nru(nru) => <dyn CoreCache<K, V>>::len(nru),
        }
    }

    /// Returns `true` if the cache contains no entries.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::builder::{CacheBuilder, CachePolicy};
    ///
    /// let mut cache = CacheBuilder::new(10).build::<u64, String>(CachePolicy::Lru);
    /// assert!(cache.is_empty());
    ///
    /// cache.insert(1, "value".to_string());
    /// assert!(!cache.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the maximum capacity of the cache.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::builder::{CacheBuilder, CachePolicy};
    ///
    /// let cache = CacheBuilder::new(100).build::<u64, String>(CachePolicy::Lru);
    /// assert_eq!(cache.capacity(), 100);
    /// ```
    pub fn capacity(&self) -> usize {
        match &self.inner {
            #[cfg(feature = "policy-fifo")]
            CacheInner::Fifo(fifo) => <dyn CoreCache<K, V>>::capacity(fifo),
            #[cfg(feature = "policy-lru")]
            CacheInner::Lru(lru) => lru.capacity(),
            #[cfg(feature = "policy-fast-lru")]
            CacheInner::FastLru(fast_lru) => fast_lru.capacity(),
            #[cfg(feature = "policy-lru-k")]
            CacheInner::LruK(lruk) => <dyn CoreCache<K, V>>::capacity(lruk),
            #[cfg(feature = "policy-lfu")]
            CacheInner::Lfu(lfu) => lfu.capacity(),
            #[cfg(feature = "policy-heap-lfu")]
            CacheInner::HeapLfu(heap_lfu) => heap_lfu.capacity(),
            #[cfg(feature = "policy-two-q")]
            CacheInner::TwoQ(twoq) => twoq.capacity(),
            #[cfg(feature = "policy-s3-fifo")]
            CacheInner::S3Fifo(s3fifo) => s3fifo.capacity(),
            #[cfg(feature = "policy-arc")]
            CacheInner::Arc(arc) => arc.capacity(),
            #[cfg(feature = "policy-lifo")]
            CacheInner::Lifo(lifo) => <dyn CoreCache<K, V>>::capacity(lifo),
            #[cfg(feature = "policy-mfu")]
            CacheInner::Mfu(mfu) => mfu.capacity(),
            #[cfg(feature = "policy-mru")]
            CacheInner::Mru(mru) => mru.capacity(),
            #[cfg(feature = "policy-random")]
            CacheInner::Random(random) => <dyn CoreCache<K, V>>::capacity(random),
            #[cfg(feature = "policy-slru")]
            CacheInner::Slru(slru) => slru.capacity(),
            #[cfg(feature = "policy-clock")]
            CacheInner::Clock(clock) => <dyn CoreCache<K, V>>::capacity(clock),
            #[cfg(feature = "policy-clock-pro")]
            CacheInner::ClockPro(clock_pro) => <dyn CoreCache<K, V>>::capacity(clock_pro),
            #[cfg(feature = "policy-nru")]
            CacheInner::Nru(nru) => <dyn CoreCache<K, V>>::capacity(nru),
        }
    }

    /// Clears all entries from the cache.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::builder::{CacheBuilder, CachePolicy};
    ///
    /// let mut cache = CacheBuilder::new(10).build::<u64, String>(CachePolicy::Lru);
    /// cache.insert(1, "one".to_string());
    /// cache.insert(2, "two".to_string());
    /// assert_eq!(cache.len(), 2);
    ///
    /// cache.clear();
    /// assert!(cache.is_empty());
    /// assert!(!cache.contains(&1));
    /// ```
    pub fn clear(&mut self) {
        match &mut self.inner {
            #[cfg(feature = "policy-fifo")]
            CacheInner::Fifo(fifo) => fifo.clear(),
            #[cfg(feature = "policy-lru")]
            CacheInner::Lru(lru) => lru.clear(),
            #[cfg(feature = "policy-fast-lru")]
            CacheInner::FastLru(fast_lru) => fast_lru.clear(),
            #[cfg(feature = "policy-lru-k")]
            CacheInner::LruK(lruk) => lruk.clear(),
            #[cfg(feature = "policy-lfu")]
            CacheInner::Lfu(lfu) => lfu.clear(),
            #[cfg(feature = "policy-heap-lfu")]
            CacheInner::HeapLfu(heap_lfu) => heap_lfu.clear(),
            #[cfg(feature = "policy-two-q")]
            CacheInner::TwoQ(twoq) => twoq.clear(),
            #[cfg(feature = "policy-s3-fifo")]
            CacheInner::S3Fifo(s3fifo) => s3fifo.clear(),
            #[cfg(feature = "policy-arc")]
            CacheInner::Arc(arc) => arc.clear(),
            #[cfg(feature = "policy-lifo")]
            CacheInner::Lifo(lifo) => lifo.clear(),
            #[cfg(feature = "policy-mfu")]
            CacheInner::Mfu(mfu) => mfu.clear(),
            #[cfg(feature = "policy-mru")]
            CacheInner::Mru(mru) => mru.clear(),
            #[cfg(feature = "policy-random")]
            CacheInner::Random(random) => random.clear(),
            #[cfg(feature = "policy-slru")]
            CacheInner::Slru(slru) => slru.clear(),
            #[cfg(feature = "policy-clock")]
            CacheInner::Clock(clock) => clock.clear(),
            #[cfg(feature = "policy-clock-pro")]
            CacheInner::ClockPro(clock_pro) => clock_pro.clear(),
            #[cfg(feature = "policy-nru")]
            CacheInner::Nru(nru) => nru.clear(),
        }
    }
}

/// Builder for creating cache instances.
///
/// # Example
///
/// ```
/// use cachekit::builder::{CacheBuilder, CachePolicy};
///
/// // Create builder with capacity
/// let builder = CacheBuilder::new(1000);
///
/// // Build different cache types from the same builder pattern
/// let lru_cache = CacheBuilder::new(100).build::<u64, String>(CachePolicy::Lru);
/// let lfu_cache = CacheBuilder::new(100).build::<u64, String>(CachePolicy::Lfu { bucket_hint: None });
/// ```
pub struct CacheBuilder {
    capacity: usize,
}

impl CacheBuilder {
    /// Creates a new cache builder with the specified capacity.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::builder::CacheBuilder;
    ///
    /// let builder = CacheBuilder::new(100);
    /// ```
    pub fn new(capacity: usize) -> Self {
        Self { capacity }
    }

    /// Build a cache with the specified policy.
    ///
    /// # Type Parameters
    ///
    /// - `K`: Key type, must be `Copy + Eq + Hash + Ord`
    /// - `V`: Value type, must be `Clone + Debug`
    ///
    /// # Example
    ///
    /// ```rust
    /// use cachekit::builder::{CacheBuilder, CachePolicy};
    ///
    /// // LRU cache
    /// let cache = CacheBuilder::new(100).build::<u64, String>(CachePolicy::Lru);
    ///
    /// // LRU-K with K=2
    /// let cache = CacheBuilder::new(100).build::<u64, String>(CachePolicy::LruK { k: 2 });
    ///
    /// // 2Q with 25% probation
    /// let cache = CacheBuilder::new(100).build::<u64, String>(CachePolicy::TwoQ { probation_frac: 0.25 });
    /// ```
    pub fn build<K, V>(self, policy: CachePolicy) -> Cache<K, V>
    where
        K: Copy + Eq + Hash + Ord,
        V: Clone + Debug,
    {
        let inner = match policy {
            #[cfg(feature = "policy-fifo")]
            CachePolicy::Fifo => CacheInner::Fifo(FifoCache::new(self.capacity)),
            #[cfg(feature = "policy-lru")]
            CachePolicy::Lru => CacheInner::Lru(LruCore::new(self.capacity)),
            #[cfg(feature = "policy-fast-lru")]
            CachePolicy::FastLru => CacheInner::FastLru(FastLru::new(self.capacity)),
            #[cfg(feature = "policy-lru-k")]
            CachePolicy::LruK { k } => CacheInner::LruK(LrukCache::with_k(self.capacity, k)),
            #[cfg(feature = "policy-lfu")]
            CachePolicy::Lfu { bucket_hint } => {
                let hint = bucket_hint.unwrap_or(DEFAULT_BUCKET_PREALLOC);
                CacheInner::Lfu(LfuCache::with_bucket_hint(self.capacity, hint))
            },
            #[cfg(feature = "policy-heap-lfu")]
            CachePolicy::HeapLfu => CacheInner::HeapLfu(HeapLfuCache::new(self.capacity)),
            #[cfg(feature = "policy-two-q")]
            CachePolicy::TwoQ { probation_frac } => {
                CacheInner::TwoQ(TwoQCore::new(self.capacity, probation_frac))
            },
            #[cfg(feature = "policy-s3-fifo")]
            CachePolicy::S3Fifo {
                small_ratio,
                ghost_ratio,
            } => CacheInner::S3Fifo(S3FifoCache::with_ratios(
                self.capacity,
                small_ratio,
                ghost_ratio,
            )),
            #[cfg(feature = "policy-arc")]
            CachePolicy::Arc => CacheInner::Arc(ARCCore::new(self.capacity)),
            #[cfg(feature = "policy-lifo")]
            CachePolicy::Lifo => CacheInner::Lifo(LifoCore::new(self.capacity)),
            #[cfg(feature = "policy-mfu")]
            CachePolicy::Mfu { bucket_hint: _ } => {
                // MfuCore uses heap internally, bucket_hint is ignored
                CacheInner::Mfu(MfuCore::new(self.capacity))
            },
            #[cfg(feature = "policy-mru")]
            CachePolicy::Mru => CacheInner::Mru(MruCore::new(self.capacity)),
            #[cfg(feature = "policy-random")]
            CachePolicy::Random => CacheInner::Random(RandomCore::new(self.capacity)),
            #[cfg(feature = "policy-slru")]
            CachePolicy::Slru { probationary_frac } => {
                CacheInner::Slru(SlruCore::new(self.capacity, probationary_frac))
            },
            #[cfg(feature = "policy-clock")]
            CachePolicy::Clock => CacheInner::Clock(ClockCache::new(self.capacity)),
            #[cfg(feature = "policy-clock-pro")]
            CachePolicy::ClockPro => CacheInner::ClockPro(ClockProCache::new(self.capacity)),
            #[cfg(feature = "policy-nru")]
            CachePolicy::Nru => CacheInner::Nru(NruCache::new(self.capacity)),
        };

        Cache { inner }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_enabled_policies() -> Vec<CachePolicy> {
        vec![
            #[cfg(feature = "policy-fifo")]
            CachePolicy::Fifo,
            #[cfg(feature = "policy-lru")]
            CachePolicy::Lru,
            #[cfg(feature = "policy-fast-lru")]
            CachePolicy::FastLru,
            #[cfg(feature = "policy-lru-k")]
            CachePolicy::LruK { k: 2 },
            #[cfg(feature = "policy-lfu")]
            CachePolicy::Lfu { bucket_hint: None },
            #[cfg(feature = "policy-heap-lfu")]
            CachePolicy::HeapLfu,
            #[cfg(feature = "policy-two-q")]
            CachePolicy::TwoQ {
                probation_frac: 0.25,
            },
            #[cfg(feature = "policy-s3-fifo")]
            CachePolicy::S3Fifo {
                small_ratio: 0.1,
                ghost_ratio: 0.9,
            },
            #[cfg(feature = "policy-arc")]
            CachePolicy::Arc,
            #[cfg(feature = "policy-lifo")]
            CachePolicy::Lifo,
            #[cfg(feature = "policy-mfu")]
            CachePolicy::Mfu { bucket_hint: None },
            #[cfg(feature = "policy-mru")]
            CachePolicy::Mru,
            #[cfg(feature = "policy-random")]
            CachePolicy::Random,
            #[cfg(feature = "policy-slru")]
            CachePolicy::Slru {
                probationary_frac: 0.25,
            },
            #[cfg(feature = "policy-clock")]
            CachePolicy::Clock,
            #[cfg(feature = "policy-clock-pro")]
            CachePolicy::ClockPro,
            #[cfg(feature = "policy-nru")]
            CachePolicy::Nru,
        ]
    }

    #[test]
    fn test_all_policies_basic_ops() {
        let policies = all_enabled_policies();
        assert!(
            !policies.is_empty(),
            "At least one policy feature must be enabled"
        );

        for policy in policies {
            let mut cache = CacheBuilder::new(10).build::<u64, String>(policy);

            // Insert
            assert_eq!(cache.insert(1, "one".to_string()), None);
            assert_eq!(cache.insert(2, "two".to_string()), None);

            // Get
            assert_eq!(cache.get(&1), Some(&"one".to_string()));
            assert_eq!(cache.get(&2), Some(&"two".to_string()));
            assert_eq!(cache.get(&3), None);

            // Contains
            assert!(cache.contains(&1));
            assert!(!cache.contains(&99));

            // Len
            assert_eq!(cache.len(), 2);
            assert!(!cache.is_empty());

            // Update
            assert_eq!(cache.insert(1, "ONE".to_string()), Some("one".to_string()));
            assert_eq!(cache.get(&1), Some(&"ONE".to_string()));

            // Clear
            cache.clear();
            assert!(cache.is_empty());
        }
    }

    #[test]
    #[cfg(feature = "policy-lru")]
    fn test_capacity_enforcement() {
        let mut cache = CacheBuilder::new(2).build::<u64, String>(CachePolicy::Lru);

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.insert(3, "three".to_string()); // Should evict key 1

        assert_eq!(cache.len(), 2);
        assert!(!cache.contains(&1)); // Evicted
        assert!(cache.contains(&2));
        assert!(cache.contains(&3));
    }
}
