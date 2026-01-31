//! # Cache Trait Hierarchy
//!
//! This module defines the trait hierarchy for the cache subsystem, providing a unified
//! interface for different cache eviction policies (FIFO, LRU, LFU, LRU-K) while ensuring
//! type safety and policy-appropriate operation sets.
//!
//! ## Architecture
//!
//! ```text
//!                          ┌─────────────────────────────────────────┐
//!                          │            CoreCache<K, V>              │
//!                          │                                         │
//!                          │  insert(&mut, K, V) → Option<V>         │
//!                          │  get(&mut, &K) → Option<&V>             │
//!                          │  contains(&, &K) → bool                 │
//!                          │  len(&) → usize                         │
//!                          │  is_empty(&) → bool                     │
//!                          │  capacity(&) → usize                    │
//!                          │  clear(&mut)                            │
//!                          └──────────────────┬──────────────────────┘
//!                                             │
//!                ┌────────────────────────────┴────────────────────────────┐
//!                │                                                         │
//!                ▼                                                         ▼
//!   ┌────────────────────────────┐                          ┌─────────────────────────────┐
//!   │   FifoCacheTrait<K, V>     │                          │    MutableCache<K, V>       │
//!   │                            │                          │                             │
//!   │  pop_oldest() → (K, V)     │                          │  remove(&K) → Option<V>     │
//!   │  peek_oldest() → (&K, &V)  │                          │  remove_batch(&[K])         │
//!   │  pop_oldest_batch(n)       │                          │                             │
//!   │  age_rank(&K) → usize      │                          └──────────────┬──────────────┘
//!   │                            │                                         │
//!   │  ⚠ No arbitrary removal!   │          ┌──────────────────────────────┼──────────────────────────────┐
//!   └────────────────────────────┘          │                              │                              │
//!                                           ▼                              ▼                              ▼
//!                          ┌────────────────────────────┐  ┌────────────────────────────┐  ┌────────────────────────────┐
//!                          │   LruCacheTrait<K, V>      │  │   LfuCacheTrait<K, V>      │  │   LrukCacheTrait<K, V>     │
//!                          │                            │  │                            │  │                            │
//!                          │  pop_lru() → (K, V)        │  │  pop_lfu() → (K, V)        │  │  pop_lru_k() → (K, V)      │
//!                          │  peek_lru() → (&K, &V)     │  │  peek_lfu() → (&K, &V)     │  │  peek_lru_k() → (&K, &V)   │
//!                          │  touch(&K) → bool          │  │  frequency(&K) → u64       │  │  k_value() → usize         │
//!                          │  recency_rank(&K) → usize  │  │  reset_frequency(&K)       │  │  access_history(&K)        │
//!                          │                            │  │  increment_frequency(&K)   │  │  access_count(&K) → usize  │
//!                          └────────────────────────────┘  └────────────────────────────┘  │  k_distance(&K) → u64      │
//!                                                                                          │  touch(&K) → bool          │
//!                                                                                          │  k_distance_rank(&K)       │
//!                                                                                          └────────────────────────────┘
//! ```
//!
//! ## Trait Design Philosophy
//!
//! ```text
//!   ┌──────────────────────────────────────────────────────────────────────────┐
//!   │                         TRAIT HIERARCHY DESIGN                           │
//!   │                                                                          │
//!   │   1. CoreCache: Universal operations ALL caches must support             │
//!   │      └── insert, get, contains, len, capacity, clear                     │
//!   │                                                                          │
//!   │   2. MutableCache: Adds arbitrary key-based removal                      │
//!   │      └── remove(&K) - NOT suitable for FIFO (breaks insertion order)     │
//!   │                                                                          │
//!   │   3. Policy-Specific Traits: Add policy-appropriate eviction             │
//!   │      ├── FIFO: pop_oldest (no arbitrary removal!)                        │
//!   │      ├── LRU:  pop_lru + touch (recency-based)                           │
//!   │      ├── LFU:  pop_lfu + frequency (frequency-based)                     │
//!   │      └── LRU-K: pop_lru_k + k_distance (scan-resistant)                  │
//!   │                                                                          │
//!   │   Key Insight: FIFO extends CoreCache directly (NOT MutableCache)        │
//!   │   because arbitrary removal would violate FIFO semantics.                │
//!   └──────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Trait Summary
//!
//! | Trait                | Extends         | Purpose                              |
//! |----------------------|-----------------|--------------------------------------|
//! | `CoreCache`          | -               | Universal cache operations           |
//! | `MutableCache`       | `CoreCache`     | Adds arbitrary key removal           |
//! | `FifoCacheTrait`     | `CoreCache`     | FIFO-specific (no remove!)           |
//! | `LruCacheTrait`      | `MutableCache`  | LRU-specific with recency tracking   |
//! | `LfuCacheTrait`      | `MutableCache`  | LFU-specific with frequency tracking |
//! | `LrukCacheTrait`     | `MutableCache`  | LRU-K with K-distance tracking       |
//! | `ConcurrentCache`    | `Send + Sync`   | Marker for thread-safe caches        |
//! | -                    | -               | -                                    |
//! | `CacheTierManager`   | -               | Multi-tier cache management          |
//! | `CacheFactory`       | -               | Cache instance creation              |
//! | `AsyncCacheFuture`   | `Send + Sync`   | Future async operation support       |
//!
//! ## Why FIFO Doesn't Extend MutableCache
//!
//! ```text
//!   FIFO Cache Semantics:
//!   ═══════════════════════════════════════════════════════════════════════════
//!
//!     VecDeque: [A] ─ [B] ─ [C] ─ [D]
//!               ↑                 ↑
//!             oldest           newest
//!
//!   If we allowed remove(&B):
//!     VecDeque: [A] ─ [C] ─ [D]   ← Order still intact, but...
//!
//!   Problem: Now VecDeque doesn't track true insertion order!
//!   - Stale entries accumulate
//!   - age_rank() becomes O(n) scanning for valid entries
//!   - FIFO semantics become muddled
//!
//!   Solution: FifoCacheTrait extends CoreCache directly, ensuring
//!   only FIFO-appropriate operations are available.
//!
//!   ═══════════════════════════════════════════════════════════════════════════
//! ```
//!
//! ## Policy Comparison
//!
//! | Policy | Eviction Basis         | Supports Remove | Best For                 |
//! |--------|------------------------|-----------------|--------------------------|
//! | FIFO   | Insertion order        | ❌ No           | Predictable eviction     |
//! | LRU    | Last access time       | ✅ Yes          | Temporal locality        |
//! | LFU    | Access frequency       | ✅ Yes          | Stable hot spots         |
//! | LRU-K  | K-th access time       | ✅ Yes          | Scan resistance          |
//!
//! ## Utility Traits
//!
//! ```text
//!   ┌─────────────────────────────────────────────────────────────────────────┐
//!   │ ConcurrentCache                                                         │
//!   │                                                                         │
//!   │   Marker trait: Send + Sync                                             │
//!   │   Purpose: Guarantee thread-safe cache implementations                  │
//!   │   Usage: fn use_cache<C: CoreCache<K, V> + ConcurrentCache>(c: &C)      │
//!   └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## CacheConfig
//!
//! | Field            | Type    | Default | Description                        |
//! |------------------|---------|---------|------------------------------------|
//! | `capacity`       | `usize` | 1000    | Maximum entries                    |
//! | `enable_stats`   | `bool`  | false   | Enable hit/miss tracking           |
//! | `prealloc_memory`| `bool`  | true    | Pre-allocate memory for capacity   |
//! | `thread_safe`    | `bool`  | false   | Use internal synchronization       |
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! use crate::storage::disk::async_disk::cache::cache_traits::{
//!     CoreCache, MutableCache, FifoCacheTrait, LruCacheTrait, LfuCacheTrait,
//! };
//!
//! // Function accepting any cache
//! fn warm_cache<C: CoreCache<u64, Vec<u8>>>(cache: &mut C, data: &[(u64, Vec<u8>)]) {
//!     for (key, value) in data {
//!         cache.insert(*key, value.clone());
//!     }
//! }
//!
//! // Function requiring removal capability (LRU, LFU - NOT FIFO)
//! fn invalidate_keys<C: MutableCache<u64, Vec<u8>>>(cache: &mut C, keys: &[u64]) {
//!     for key in keys {
//!         cache.remove(key);
//!     }
//! }
//!
//! // FIFO-specific function
//! fn evict_oldest_batch<C: FifoCacheTrait<u64, Vec<u8>>>(
//!     cache: &mut C,
//!     count: usize,
//! ) -> Vec<(u64, Vec<u8>)> {
//!     cache.pop_oldest_batch(count)
//! }
//!
//! // LRU-specific function
//! fn touch_hot_keys<C: LruCacheTrait<u64, Vec<u8>>>(cache: &mut C, keys: &[u64]) {
//!     for key in keys {
//!         cache.touch(key); // Mark as recently used without retrieving
//!     }
//! }
//!
//! // LFU-specific function with frequency-based prioritization
//! fn boost_key_priority<C: LfuCacheTrait<u64, Vec<u8>>>(cache: &mut C, key: &u64) {
//!     // Increment frequency without accessing value
//!     cache.increment_frequency(key);
//! }
//!
//! // Thread-safe cache usage
//! use std::sync::{Arc, RwLock};
//! use crate::storage::disk::async_disk::cache::lru::ConcurrentLruCache;
//!
//! let shared_cache = Arc::new(ConcurrentLruCache::<u64, Vec<u8>>::new(1000));
//!
//! // Safe to use from multiple threads
//! let cache_clone = shared_cache.clone();
//! std::thread::spawn(move || {
//!     cache_clone.insert(42, vec![1, 2, 3]);
//! });
//! ```
//!
//! ## Thread Safety
//!
//! - Individual cache implementations are **NOT thread-safe** by default
//! - Use `ConcurrentCache` marker trait to identify thread-safe implementations
//! - Wrap non-concurrent caches in `Arc<RwLock<C>>` for shared access
//! - Some implementations (e.g., `ConcurrentLruCache`) provide built-in concurrency
//!
//! ## Implementation Notes
//!
//! - **Trait Bounds**: `CoreCache` has no bounds on K, V; implementations add as needed
//! - **Default Implementations**: `is_empty()`, `total_misses()`, `remove_batch()`, `pop_oldest_batch()`
//! - **Batch Operations**: Default implementations loop over single operations
//! - **Async Support**: `AsyncCacheFuture` prepared for Phase 2 async-trait integration

/// Core cache operations that all caches support.
///
/// This trait defines the fundamental operations that make sense for any cache type,
/// regardless of eviction policy. All policy-specific traits extend this.
///
/// # Type Parameters
///
/// - `K`: Key type (implementations typically require `Eq + Hash`)
/// - `V`: Value type
///
/// # Example
///
/// ```
/// use cachekit::traits::CoreCache;
/// use cachekit::policy::lru_k::LrukCache;
///
/// fn warm_cache<C: CoreCache<u64, String>>(cache: &mut C, data: &[(u64, String)]) {
///     for (key, value) in data {
///         cache.insert(*key, value.clone());
///     }
/// }
///
/// let mut cache = LrukCache::new(100);
/// warm_cache(&mut cache, &[(1, "one".to_string()), (2, "two".to_string())]);
/// assert_eq!(cache.len(), 2);
/// ```
pub trait CoreCache<K, V> {
    /// Inserts a key-value pair, returning the previous value if it existed.
    ///
    /// If the cache is at capacity, an entry may be evicted according to the
    /// cache's eviction policy before the new entry is inserted.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::traits::CoreCache;
    /// use cachekit::policy::lru_k::LrukCache;
    ///
    /// let mut cache = LrukCache::new(10);
    ///
    /// // New key returns None
    /// assert_eq!(cache.insert(1, "first"), None);
    ///
    /// // Existing key returns previous value
    /// assert_eq!(cache.insert(1, "second"), Some("first"));
    /// ```
    fn insert(&mut self, key: K, value: V) -> Option<V>;

    /// Gets a reference to a value by key.
    ///
    /// May update internal state (access time, frequency) depending on the
    /// eviction policy. Use [`contains`](Self::contains) if you only need
    /// to check existence without affecting eviction order.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::traits::CoreCache;
    /// use cachekit::policy::lru_k::LrukCache;
    ///
    /// let mut cache = LrukCache::new(10);
    /// cache.insert(1, "value");
    ///
    /// assert_eq!(cache.get(&1), Some(&"value"));
    /// assert_eq!(cache.get(&99), None);
    /// ```
    fn get(&mut self, key: &K) -> Option<&V>;

    /// Checks if a key exists without updating access state.
    ///
    /// Unlike [`get`](Self::get), this does not affect eviction order.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::traits::CoreCache;
    /// use cachekit::policy::lru_k::LrukCache;
    ///
    /// let mut cache = LrukCache::new(10);
    /// cache.insert(1, "value");
    ///
    /// assert!(cache.contains(&1));
    /// assert!(!cache.contains(&99));
    /// ```
    fn contains(&self, key: &K) -> bool;

    /// Returns the current number of entries in the cache.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::traits::CoreCache;
    /// use cachekit::policy::lru_k::LrukCache;
    ///
    /// let mut cache = LrukCache::new(10);
    /// assert_eq!(cache.len(), 0);
    ///
    /// cache.insert(1, "one");
    /// cache.insert(2, "two");
    /// assert_eq!(cache.len(), 2);
    /// ```
    fn len(&self) -> usize;

    /// Returns `true` if the cache contains no entries.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::traits::CoreCache;
    /// use cachekit::policy::lru_k::LrukCache;
    ///
    /// let mut cache: LrukCache<u64, &str> = LrukCache::new(10);
    /// assert!(cache.is_empty());
    ///
    /// cache.insert(1, "value");
    /// assert!(!cache.is_empty());
    /// ```
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the maximum capacity of the cache.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::traits::CoreCache;
    /// use cachekit::policy::lru_k::LrukCache;
    ///
    /// let cache: LrukCache<u64, &str> = LrukCache::new(100);
    /// assert_eq!(cache.capacity(), 100);
    /// ```
    fn capacity(&self) -> usize;

    /// Removes all entries from the cache.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::traits::CoreCache;
    /// use cachekit::policy::lru_k::LrukCache;
    ///
    /// let mut cache = LrukCache::new(10);
    /// cache.insert(1, "one");
    /// cache.insert(2, "two");
    /// assert_eq!(cache.len(), 2);
    ///
    /// cache.clear();
    /// assert!(cache.is_empty());
    /// ```
    fn clear(&mut self);
}

/// Caches that support arbitrary key-based removal.
///
/// This trait extends [`CoreCache`] with the ability to remove entries by key.
/// Appropriate for LRU, LFU, and general hash-map style caches where arbitrary
/// removal doesn't violate policy semantics.
///
/// **Note**: FIFO caches intentionally do NOT implement this trait because
/// arbitrary removal would violate FIFO semantics. Use [`FifoCacheTrait`] instead.
///
/// # Example
///
/// ```
/// use cachekit::traits::{CoreCache, MutableCache};
/// use cachekit::policy::lru_k::LrukCache;
///
/// fn invalidate_keys<C: MutableCache<u64, String>>(cache: &mut C, keys: &[u64]) {
///     for key in keys {
///         cache.remove(key);
///     }
/// }
///
/// let mut cache = LrukCache::new(100);
/// cache.insert(1, "one".to_string());
/// cache.insert(2, "two".to_string());
/// cache.insert(3, "three".to_string());
///
/// invalidate_keys(&mut cache, &[1, 3]);
/// assert!(!cache.contains(&1));
/// assert!(cache.contains(&2));
/// assert!(!cache.contains(&3));
/// ```
pub trait MutableCache<K, V>: CoreCache<K, V> {
    /// Removes a specific key-value pair.
    ///
    /// Returns the removed value if the key existed, or `None` if it didn't.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::traits::{CoreCache, MutableCache};
    /// use cachekit::policy::lru_k::LrukCache;
    ///
    /// let mut cache = LrukCache::new(10);
    /// cache.insert(1, "value");
    ///
    /// assert_eq!(cache.remove(&1), Some("value"));
    /// assert_eq!(cache.remove(&1), None);  // Already removed
    /// ```
    fn remove(&mut self, key: &K) -> Option<V>;

    /// Removes multiple keys efficiently.
    ///
    /// Returns a vector of `Option<V>` in the same order as the input keys.
    /// The default implementation loops over [`remove`](Self::remove).
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::traits::{CoreCache, MutableCache};
    /// use cachekit::policy::lru_k::LrukCache;
    ///
    /// let mut cache = LrukCache::new(10);
    /// cache.insert(1, "one");
    /// cache.insert(2, "two");
    /// cache.insert(3, "three");
    ///
    /// let removed = cache.remove_batch(&[1, 99, 3]);
    /// assert_eq!(removed, vec![Some("one"), None, Some("three")]);
    /// assert_eq!(cache.len(), 1);
    /// ```
    fn remove_batch(&mut self, keys: &[K]) -> Vec<Option<V>> {
        keys.iter().map(|k| self.remove(k)).collect()
    }
}

/// FIFO-specific operations that respect insertion order.
///
/// This trait extends [`CoreCache`] with FIFO-appropriate operations.
/// Importantly, it does NOT extend [`MutableCache`] because arbitrary removal
/// would violate FIFO semantics (insertion order tracking).
///
/// # Design Rationale
///
/// FIFO caches evict in insertion order. If we allowed `remove(&key)`:
/// - The queue would have "holes"
/// - `age_rank()` would need expensive O(n) scanning
/// - True insertion order would be lost
///
/// # Example
///
/// ```
/// use cachekit::traits::{CoreCache, FifoCacheTrait};
/// use cachekit::policy::fifo::FifoCache;
///
/// let mut cache = FifoCache::new(3);
/// cache.insert(1, "first");
/// cache.insert(2, "second");
/// cache.insert(3, "third");
///
/// // Peek without removing
/// assert_eq!(cache.peek_oldest(), Some((&1, &"first")));
///
/// // Pop oldest entry
/// assert_eq!(cache.pop_oldest(), Some((1, "first")));
/// assert_eq!(cache.len(), 2);
///
/// // Age rank (0 = oldest)
/// assert_eq!(cache.age_rank(&2), Some(0));  // Now oldest
/// assert_eq!(cache.age_rank(&3), Some(1));
/// ```
pub trait FifoCacheTrait<K, V>: CoreCache<K, V> {
    /// Removes and returns the oldest entry (first inserted).
    ///
    /// Returns `None` if the cache is empty.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::traits::{CoreCache, FifoCacheTrait};
    /// use cachekit::policy::fifo::FifoCache;
    ///
    /// let mut cache = FifoCache::new(10);
    /// cache.insert(1, "first");
    /// cache.insert(2, "second");
    ///
    /// assert_eq!(cache.pop_oldest(), Some((1, "first")));
    /// assert_eq!(cache.pop_oldest(), Some((2, "second")));
    /// assert_eq!(cache.pop_oldest(), None);
    /// ```
    fn pop_oldest(&mut self) -> Option<(K, V)>;

    /// Peeks at the oldest entry without removing it.
    ///
    /// Returns `None` if the cache is empty.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::traits::{CoreCache, FifoCacheTrait};
    /// use cachekit::policy::fifo::FifoCache;
    ///
    /// let mut cache = FifoCache::new(10);
    /// cache.insert(1, "first");
    ///
    /// // Peek doesn't remove
    /// assert_eq!(cache.peek_oldest(), Some((&1, &"first")));
    /// assert_eq!(cache.peek_oldest(), Some((&1, &"first")));
    /// assert_eq!(cache.len(), 1);
    /// ```
    fn peek_oldest(&self) -> Option<(&K, &V)>;

    /// Removes multiple oldest entries efficiently.
    ///
    /// Returns up to `count` entries in FIFO order (oldest first).
    /// The default implementation calls [`pop_oldest`](Self::pop_oldest) in a loop.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::traits::{CoreCache, FifoCacheTrait};
    /// use cachekit::policy::fifo::FifoCache;
    ///
    /// let mut cache = FifoCache::new(10);
    /// cache.insert(1, "a");
    /// cache.insert(2, "b");
    /// cache.insert(3, "c");
    ///
    /// let batch = cache.pop_oldest_batch(2);
    /// assert_eq!(batch, vec![(1, "a"), (2, "b")]);
    /// assert_eq!(cache.len(), 1);
    /// ```
    fn pop_oldest_batch(&mut self, count: usize) -> Vec<(K, V)> {
        (0..count).filter_map(|_| self.pop_oldest()).collect()
    }

    /// Gets the age rank of a key (0 = oldest, higher = newer).
    ///
    /// Returns `None` if the key is not found.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::traits::{CoreCache, FifoCacheTrait};
    /// use cachekit::policy::fifo::FifoCache;
    ///
    /// let mut cache = FifoCache::new(10);
    /// cache.insert(1, "first");
    /// cache.insert(2, "second");
    /// cache.insert(3, "third");
    ///
    /// assert_eq!(cache.age_rank(&1), Some(0));  // Oldest
    /// assert_eq!(cache.age_rank(&2), Some(1));
    /// assert_eq!(cache.age_rank(&3), Some(2));  // Newest
    /// assert_eq!(cache.age_rank(&99), None);    // Not found
    /// ```
    fn age_rank(&self, key: &K) -> Option<usize>;
}

/// LRU-specific operations that respect access order.
///
/// This trait extends [`MutableCache`] with LRU-specific eviction and access
/// tracking operations. Entries are ordered by recency—the least recently
/// accessed entry is evicted first.
///
/// # Example
///
/// ```
/// use std::sync::Arc;
/// use cachekit::traits::{CoreCache, MutableCache, LruCacheTrait};
/// use cachekit::policy::lru::LruCore;
///
/// let mut cache: LruCore<u64, &str> = LruCore::new(3);
/// cache.insert(1, Arc::new("first"));
/// cache.insert(2, Arc::new("second"));
/// cache.insert(3, Arc::new("third"));
///
/// // Access key 1 to make it MRU
/// cache.get(&1);
///
/// // Key 2 is now LRU
/// assert_eq!(cache.peek_lru().map(|(k, _)| *k), Some(2));
///
/// // Touch without retrieving value
/// assert!(cache.touch(&2));  // Now key 3 is LRU
///
/// // Pop LRU entry
/// let (key, _) = cache.pop_lru().unwrap();
/// assert_eq!(key, 3);
/// ```
pub trait LruCacheTrait<K, V>: MutableCache<K, V> {
    /// Removes and returns the least recently used entry.
    ///
    /// Returns `None` if the cache is empty.
    ///
    /// # Example
    ///
    /// ```
    /// use std::sync::Arc;
    /// use cachekit::traits::{CoreCache, LruCacheTrait};
    /// use cachekit::policy::lru::LruCore;
    ///
    /// let mut cache: LruCore<u64, &str> = LruCore::new(10);
    /// cache.insert(1, Arc::new("first"));
    /// cache.insert(2, Arc::new("second"));
    ///
    /// let (key, _) = cache.pop_lru().unwrap();
    /// assert_eq!(key, 1);  // First inserted, not accessed since
    /// ```
    fn pop_lru(&mut self) -> Option<(K, V)>;

    /// Peeks at the LRU entry without removing it.
    ///
    /// Returns `None` if the cache is empty. Does not update access time.
    ///
    /// # Example
    ///
    /// ```
    /// use std::sync::Arc;
    /// use cachekit::traits::{CoreCache, LruCacheTrait};
    /// use cachekit::policy::lru::LruCore;
    ///
    /// let mut cache: LruCore<u64, &str> = LruCore::new(10);
    /// cache.insert(1, Arc::new("first"));
    /// cache.insert(2, Arc::new("second"));
    ///
    /// // Peek doesn't affect order
    /// assert_eq!(cache.peek_lru().map(|(k, _)| *k), Some(1));
    /// assert_eq!(cache.peek_lru().map(|(k, _)| *k), Some(1));
    /// ```
    fn peek_lru(&self) -> Option<(&K, &V)>;

    /// Marks an entry as recently used without retrieving the value.
    ///
    /// Returns `true` if the key was found and touched, `false` otherwise.
    /// This is useful for refreshing eviction order without fetching data.
    ///
    /// # Example
    ///
    /// ```
    /// use std::sync::Arc;
    /// use cachekit::traits::{CoreCache, LruCacheTrait};
    /// use cachekit::policy::lru::LruCore;
    ///
    /// let mut cache: LruCore<u64, &str> = LruCore::new(10);
    /// cache.insert(1, Arc::new("first"));
    /// cache.insert(2, Arc::new("second"));
    ///
    /// // Key 1 is LRU
    /// assert_eq!(cache.peek_lru().map(|(k, _)| *k), Some(1));
    ///
    /// // Touch key 1 to make it MRU
    /// assert!(cache.touch(&1));
    ///
    /// // Now key 2 is LRU
    /// assert_eq!(cache.peek_lru().map(|(k, _)| *k), Some(2));
    ///
    /// // Touch non-existent key returns false
    /// assert!(!cache.touch(&99));
    /// ```
    fn touch(&mut self, key: &K) -> bool;

    /// Gets the recency rank of a key (0 = most recent, higher = less recent).
    ///
    /// Returns `None` if the key is not found.
    ///
    /// # Example
    ///
    /// ```
    /// use std::sync::Arc;
    /// use cachekit::traits::{CoreCache, LruCacheTrait};
    /// use cachekit::policy::lru::LruCore;
    ///
    /// let mut cache: LruCore<u64, &str> = LruCore::new(10);
    /// cache.insert(1, Arc::new("first"));
    /// cache.insert(2, Arc::new("second"));
    /// cache.insert(3, Arc::new("third"));
    ///
    /// // Most recent insertion is rank 0
    /// assert_eq!(cache.recency_rank(&3), Some(0));
    /// assert_eq!(cache.recency_rank(&2), Some(1));
    /// assert_eq!(cache.recency_rank(&1), Some(2));  // Oldest
    /// assert_eq!(cache.recency_rank(&99), None);
    /// ```
    fn recency_rank(&self, key: &K) -> Option<usize>;
}

/// LFU-specific operations that respect frequency order.
///
/// This trait extends [`MutableCache`] with LFU-specific eviction and frequency
/// tracking operations. Entries are ordered by access frequency—the least
/// frequently accessed entry is evicted first.
///
/// # Example
///
/// ```
/// use std::sync::Arc;
/// use cachekit::traits::{CoreCache, MutableCache, LfuCacheTrait};
/// use cachekit::policy::lfu::LfuCache;
///
/// let mut cache: LfuCache<u64, &str> = LfuCache::new(3);
/// cache.insert(1, Arc::new("first"));
/// cache.insert(2, Arc::new("second"));
/// cache.insert(3, Arc::new("third"));
///
/// // Access key 1 multiple times
/// cache.get(&1);
/// cache.get(&1);
/// cache.get(&1);
///
/// // Key 1 now has frequency 4 (1 insert + 3 gets)
/// assert_eq!(cache.frequency(&1), Some(4));
///
/// // Key 2 has frequency 1 (just insert)
/// assert_eq!(cache.frequency(&2), Some(1));
///
/// // Pop LFU (key 2 or 3, both have freq=1)
/// let (key, _) = cache.pop_lfu().unwrap();
/// assert!(key == 2 || key == 3);
/// ```
pub trait LfuCacheTrait<K, V>: MutableCache<K, V> {
    /// Removes and returns the least frequently used entry.
    ///
    /// If multiple entries have the same frequency, eviction order depends
    /// on the implementation (typically FIFO among same-frequency entries).
    /// Returns `None` if the cache is empty.
    ///
    /// # Example
    ///
    /// ```
    /// use std::sync::Arc;
    /// use cachekit::traits::{CoreCache, LfuCacheTrait};
    /// use cachekit::policy::lfu::LfuCache;
    ///
    /// let mut cache: LfuCache<u64, &str> = LfuCache::new(10);
    /// cache.insert(1, Arc::new("first"));
    /// cache.insert(2, Arc::new("second"));
    ///
    /// // Access key 2 to increase its frequency
    /// cache.get(&2);
    ///
    /// // Key 1 is LFU (freq=1 vs freq=2)
    /// let (key, _) = cache.pop_lfu().unwrap();
    /// assert_eq!(key, 1);
    /// ```
    fn pop_lfu(&mut self) -> Option<(K, V)>;

    /// Peeks at the LFU entry without removing it.
    ///
    /// Returns `None` if the cache is empty. Does not increment frequency.
    ///
    /// # Example
    ///
    /// ```
    /// use std::sync::Arc;
    /// use cachekit::traits::{CoreCache, LfuCacheTrait};
    /// use cachekit::policy::lfu::LfuCache;
    ///
    /// let mut cache: LfuCache<u64, &str> = LfuCache::new(10);
    /// cache.insert(1, Arc::new("first"));
    /// cache.insert(2, Arc::new("second"));
    /// cache.get(&2);  // freq=2
    ///
    /// // Key 1 is LFU
    /// assert_eq!(cache.peek_lfu().map(|(k, _)| *k), Some(1));
    /// ```
    fn peek_lfu(&self) -> Option<(&K, &V)>;

    /// Gets the access frequency for a key.
    ///
    /// Returns `None` if the key is not found.
    ///
    /// # Example
    ///
    /// ```
    /// use std::sync::Arc;
    /// use cachekit::traits::{CoreCache, LfuCacheTrait};
    /// use cachekit::policy::lfu::LfuCache;
    ///
    /// let mut cache: LfuCache<u64, &str> = LfuCache::new(10);
    /// cache.insert(1, Arc::new("value"));
    /// assert_eq!(cache.frequency(&1), Some(1));
    ///
    /// cache.get(&1);
    /// assert_eq!(cache.frequency(&1), Some(2));
    ///
    /// assert_eq!(cache.frequency(&99), None);
    /// ```
    fn frequency(&self, key: &K) -> Option<u64>;

    /// Resets the frequency counter for a key to 1.
    ///
    /// Returns the old frequency if the key existed, `None` otherwise.
    /// Useful for demoting hot entries after access pattern changes.
    ///
    /// # Example
    ///
    /// ```
    /// use std::sync::Arc;
    /// use cachekit::traits::{CoreCache, LfuCacheTrait};
    /// use cachekit::policy::lfu::LfuCache;
    ///
    /// let mut cache: LfuCache<u64, &str> = LfuCache::new(10);
    /// cache.insert(1, Arc::new("value"));
    /// cache.get(&1);
    /// cache.get(&1);
    /// assert_eq!(cache.frequency(&1), Some(3));
    ///
    /// // Reset to 1
    /// assert_eq!(cache.reset_frequency(&1), Some(3));
    /// assert_eq!(cache.frequency(&1), Some(1));
    /// ```
    fn reset_frequency(&mut self, key: &K) -> Option<u64>;

    /// Increments frequency without accessing the value.
    ///
    /// Returns the new frequency if the key existed, `None` otherwise.
    /// Useful for boosting priority without triggering value access.
    ///
    /// # Example
    ///
    /// ```
    /// use std::sync::Arc;
    /// use cachekit::traits::{CoreCache, LfuCacheTrait};
    /// use cachekit::policy::lfu::LfuCache;
    ///
    /// let mut cache: LfuCache<u64, &str> = LfuCache::new(10);
    /// cache.insert(1, Arc::new("value"));
    /// assert_eq!(cache.frequency(&1), Some(1));
    ///
    /// // Boost without accessing
    /// assert_eq!(cache.increment_frequency(&1), Some(2));
    /// assert_eq!(cache.increment_frequency(&1), Some(3));
    ///
    /// assert_eq!(cache.increment_frequency(&99), None);
    /// ```
    fn increment_frequency(&mut self, key: &K) -> Option<u64>;
}

/// LRU-K specific operations that respect K-distance access patterns.
///
/// This trait extends [`MutableCache`] with LRU-K-specific eviction and access
/// history tracking. Unlike standard LRU which considers only the last access,
/// LRU-K tracks the K-th most recent access time, providing scan resistance.
///
/// # Scan Resistance
///
/// LRU-K protects the cache from pollution by one-time scans. An entry needs
/// K accesses before it can displace frequently-accessed entries.
///
/// # Example
///
/// ```
/// use cachekit::traits::{CoreCache, MutableCache, LrukCacheTrait};
/// use cachekit::policy::lru_k::LrukCache;
///
/// // Create LRU-2 cache (K=2)
/// let mut cache = LrukCache::with_k(100, 2);
/// cache.insert(1, "value");
///
/// // After insert, access_count is 1
/// assert_eq!(cache.access_count(&1), Some(1));
///
/// // No K-distance yet (need K=2 accesses)
/// assert_eq!(cache.k_distance(&1), None);
///
/// // Second access establishes K-distance
/// cache.get(&1);
/// assert_eq!(cache.access_count(&1), Some(2));
/// assert!(cache.k_distance(&1).is_some());
///
/// // Access history (most recent first)
/// let history = cache.access_history(&1).unwrap();
/// assert_eq!(history.len(), 2);
/// ```
pub trait LrukCacheTrait<K, V>: MutableCache<K, V> {
    /// Removes and returns the entry with the oldest K-th access time.
    ///
    /// Entries with fewer than K accesses are evicted first (cold entries).
    /// Returns `None` if the cache is empty.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::traits::{CoreCache, LrukCacheTrait};
    /// use cachekit::policy::lru_k::LrukCache;
    ///
    /// let mut cache = LrukCache::with_k(10, 2);
    /// cache.insert(1, "first");
    /// cache.insert(2, "second");
    ///
    /// // Access key 2 twice (makes it "hot")
    /// cache.get(&2);
    ///
    /// // Key 1 is evicted first (only 1 access, K=2 not reached)
    /// let (key, _) = cache.pop_lru_k().unwrap();
    /// assert_eq!(key, 1);
    /// ```
    fn pop_lru_k(&mut self) -> Option<(K, V)>;

    /// Peeks at the LRU-K entry without removing it.
    ///
    /// Returns `None` if the cache is empty. Does not update access history.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::traits::{CoreCache, LrukCacheTrait};
    /// use cachekit::policy::lru_k::LrukCache;
    ///
    /// let mut cache = LrukCache::with_k(10, 2);
    /// cache.insert(1, "first");
    /// cache.insert(2, "second");
    /// cache.get(&2);  // Second access for key 2
    ///
    /// // Key 1 is LRU-K (cold, only 1 access)
    /// assert_eq!(cache.peek_lru_k().map(|(k, _)| *k), Some(1));
    /// ```
    fn peek_lru_k(&self) -> Option<(&K, &V)>;

    /// Gets the K value used by this cache.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::traits::LrukCacheTrait;
    /// use cachekit::policy::lru_k::LrukCache;
    ///
    /// let cache: LrukCache<u64, &str> = LrukCache::with_k(100, 3);
    /// assert_eq!(cache.k_value(), 3);
    ///
    /// // Default K=2
    /// let default_cache: LrukCache<u64, &str> = LrukCache::new(100);
    /// assert_eq!(default_cache.k_value(), 2);
    /// ```
    fn k_value(&self) -> usize;

    /// Gets the access history for a key (most recent first).
    ///
    /// Returns up to K timestamps. Returns `None` if key not found.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::traits::{CoreCache, LrukCacheTrait};
    /// use cachekit::policy::lru_k::LrukCache;
    ///
    /// let mut cache = LrukCache::with_k(10, 3);
    /// cache.insert(1, "value");
    /// cache.get(&1);
    /// cache.get(&1);
    ///
    /// let history = cache.access_history(&1).unwrap();
    /// assert_eq!(history.len(), 3);  // 1 insert + 2 gets, up to K=3
    /// // history[0] is most recent, history[2] is oldest
    /// ```
    fn access_history(&self, key: &K) -> Option<Vec<u64>>;

    /// Gets the number of recorded accesses for a key.
    ///
    /// Returns `None` if the key is not found.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::traits::{CoreCache, LrukCacheTrait};
    /// use cachekit::policy::lru_k::LrukCache;
    ///
    /// let mut cache = LrukCache::with_k(10, 5);
    /// cache.insert(1, "value");
    /// assert_eq!(cache.access_count(&1), Some(1));
    ///
    /// cache.get(&1);
    /// cache.get(&1);
    /// assert_eq!(cache.access_count(&1), Some(3));
    ///
    /// // Capped at K
    /// cache.get(&1);
    /// cache.get(&1);
    /// cache.get(&1);
    /// assert_eq!(cache.access_count(&1), Some(5));  // Max K=5
    /// ```
    fn access_count(&self, key: &K) -> Option<usize>;

    /// Gets the K-th most recent access time for a key.
    ///
    /// Returns `None` if the key is not found or has fewer than K accesses.
    /// This is the core metric for LRU-K eviction decisions.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::traits::{CoreCache, LrukCacheTrait};
    /// use cachekit::policy::lru_k::LrukCache;
    ///
    /// let mut cache = LrukCache::with_k(10, 2);
    /// cache.insert(1, "value");
    ///
    /// // Only 1 access, no K-distance yet
    /// assert_eq!(cache.k_distance(&1), None);
    ///
    /// // Second access establishes K-distance
    /// cache.get(&1);
    /// assert!(cache.k_distance(&1).is_some());
    /// ```
    fn k_distance(&self, key: &K) -> Option<u64>;

    /// Records an access without retrieving the value.
    ///
    /// Returns `true` if the key was found and touched, `false` otherwise.
    /// This updates the access history for the entry.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::traits::{CoreCache, LrukCacheTrait};
    /// use cachekit::policy::lru_k::LrukCache;
    ///
    /// let mut cache = LrukCache::with_k(10, 2);
    /// cache.insert(1, "value");
    /// assert_eq!(cache.access_count(&1), Some(1));
    ///
    /// // Touch to record access
    /// assert!(cache.touch(&1));
    /// assert_eq!(cache.access_count(&1), Some(2));
    ///
    /// // Touch non-existent key
    /// assert!(!cache.touch(&99));
    /// ```
    fn touch(&mut self, key: &K) -> bool;

    /// Gets the rank of a key based on K-distance.
    ///
    /// Lower rank (0) means oldest K-distance (first to be evicted).
    /// Entries with fewer than K accesses are ranked by their earliest access time.
    /// Returns `None` if key not found.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::traits::{CoreCache, LrukCacheTrait};
    /// use cachekit::policy::lru_k::LrukCache;
    ///
    /// let mut cache = LrukCache::with_k(10, 2);
    /// cache.insert(1, "first");
    /// cache.insert(2, "second");
    ///
    /// // Both have 1 access (cold), ranked by insertion order
    /// assert_eq!(cache.k_distance_rank(&1), Some(0));  // Oldest
    /// assert_eq!(cache.k_distance_rank(&2), Some(1));
    /// ```
    fn k_distance_rank(&self, key: &K) -> Option<usize>;
}

/// Marker trait for caches that are safe to use concurrently.
///
/// Implementors guarantee thread-safe operations. This trait extends
/// `Send + Sync` and can be used as a bound to require concurrent access.
///
/// # Example
///
/// ```
/// use cachekit::traits::{CoreCache, ConcurrentCache};
///
/// // Function requiring a thread-safe cache
/// fn use_from_threads<K, V, C>(cache: &C)
/// where
///     K: Send + Sync,
///     V: Send + Sync,
///     C: CoreCache<K, V> + ConcurrentCache,
/// {
///     // Safe to share between threads
/// }
/// ```
///
/// # Thread Safety
///
/// Individual cache implementations are NOT thread-safe by default.
/// To use a non-concurrent cache from multiple threads, wrap it:
///
/// ```
/// use std::sync::{Arc, RwLock};
/// use cachekit::traits::CoreCache;
/// use cachekit::policy::lru_k::LrukCache;
///
/// let cache = Arc::new(RwLock::new(LrukCache::<u64, String>::new(100)));
///
/// // Clone for use in another thread
/// let cache_clone = cache.clone();
/// std::thread::spawn(move || {
///     let mut guard = cache_clone.write().unwrap();
///     guard.insert(1, "value".to_string());
/// });
/// ```
pub trait ConcurrentCache: Send + Sync {}

/// High-level cache tier management.
///
/// This trait defines a multi-tier cache architecture where entries can be
/// promoted or demoted between tiers based on access patterns:
///
/// - **Hot tier**: Frequently accessed data (LRU-managed)
/// - **Warm tier**: Moderately accessed data (LFU-managed)
/// - **Cold tier**: Rarely accessed data (FIFO-managed)
///
/// # Architecture
///
/// ```text
/// ┌──────────────┐    promote()    ┌──────────────┐    promote()    ┌──────────────┐
/// │  Cold Tier   │ ───────────────►│  Warm Tier   │───────────────► │   Hot Tier   │
/// │  (FIFO)      │                 │  (LFU)       │                 │   (LRU)      │
/// │              │◄─────────────── │              │◄───────────────  │              │
/// └──────────────┘    demote()     └──────────────┘    demote()     └──────────────┘
/// ```
///
/// # Associated Types
///
/// - `HotCache`: LRU-based cache for frequently accessed data
/// - `WarmCache`: LFU-based cache for moderately accessed data
/// - `ColdCache`: FIFO-based cache for cold/new data
pub trait CacheTierManager<K, V> {
    /// LRU-based cache for hot (frequently accessed) data.
    type HotCache: LruCacheTrait<K, V> + ConcurrentCache;

    /// LFU-based cache for warm (moderately accessed) data.
    type WarmCache: LfuCacheTrait<K, V> + ConcurrentCache;

    /// FIFO-based cache for cold (rarely accessed) data.
    type ColdCache: FifoCacheTrait<K, V> + ConcurrentCache;

    /// Promotes an entry from a lower tier to a higher tier.
    ///
    /// Returns `true` if the promotion was successful, `false` if the key
    /// wasn't found in the source tier.
    fn promote(&mut self, key: &K, from_tier: CacheTier, to_tier: CacheTier) -> bool;

    /// Demotes an entry from a higher tier to a lower tier.
    ///
    /// Returns `true` if the demotion was successful, `false` if the key
    /// wasn't found in the source tier.
    fn demote(&mut self, key: &K, from_tier: CacheTier, to_tier: CacheTier) -> bool;

    /// Gets the tier where a key currently resides.
    ///
    /// Returns `None` if the key is not in any tier.
    fn locate_key(&self, key: &K) -> Option<CacheTier>;

    /// Forces eviction from a specific tier.
    ///
    /// Returns the evicted entry, or `None` if the tier is empty.
    fn evict_from_tier(&mut self, tier: CacheTier) -> Option<(K, V)>;
}

/// Cache tier enumeration for multi-tier cache architectures.
///
/// Used with [`CacheTierManager`] to specify which tier to promote to,
/// demote from, or query.
///
/// # Tier Characteristics
///
/// | Tier | Access Pattern | Eviction Policy | Typical Use |
/// |------|----------------|-----------------|-------------|
/// | Hot  | Frequent       | LRU             | Active working set |
/// | Warm | Moderate       | LFU             | Periodically accessed |
/// | Cold | Rare           | FIFO            | New or stale data |
///
/// # Example
///
/// ```
/// use cachekit::traits::CacheTier;
///
/// let tier = CacheTier::Hot;
/// assert_eq!(tier, CacheTier::Hot);
///
/// // Tiers can be compared and hashed
/// use std::collections::HashSet;
/// let mut tiers = HashSet::new();
/// tiers.insert(CacheTier::Hot);
/// tiers.insert(CacheTier::Warm);
/// tiers.insert(CacheTier::Cold);
/// assert_eq!(tiers.len(), 3);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CacheTier {
    /// Hot tier: frequently accessed data (LRU-managed).
    ///
    /// Best for: active working set, recently accessed entries.
    Hot,

    /// Warm tier: moderately accessed data (LFU-managed).
    ///
    /// Best for: periodically accessed entries, stable hot spots.
    Warm,

    /// Cold tier: rarely accessed data (FIFO-managed).
    ///
    /// Best for: new entries, infrequently accessed data.
    Cold,
}

/// Factory trait for creating cache instances.
///
/// Provides a standard interface for cache construction, allowing generic
/// code to create cache instances without knowing the concrete type.
///
/// # Associated Types
///
/// - `Cache`: The concrete cache type produced by this factory
///
/// # Example
///
/// ```ignore
/// use cachekit::traits::{CoreCache, CacheFactory, CacheConfig};
///
/// struct LruFactory;
///
/// impl CacheFactory<u64, String> for LruFactory {
///     type Cache = LruCache<u64, String>;
///
///     fn create(capacity: usize) -> Self::Cache {
///         LruCache::new(capacity)
///     }
///
///     fn create_with_config(config: CacheConfig) -> Self::Cache {
///         LruCache::new(config.capacity)
///     }
/// }
///
/// // Generic function using factory
/// fn build_cache<F: CacheFactory<u64, String>>() -> F::Cache {
///     F::create(100)
/// }
/// ```
pub trait CacheFactory<K, V> {
    /// The concrete cache type produced by this factory.
    type Cache: CoreCache<K, V>;

    /// Creates a new cache instance with the specified capacity.
    fn create(capacity: usize) -> Self::Cache;

    /// Creates a cache with custom configuration.
    fn create_with_config(config: CacheConfig) -> Self::Cache;
}

/// Configuration for cache creation.
///
/// Used with [`CacheFactory::create_with_config`] to customize cache behavior.
///
/// # Fields
///
/// | Field | Type | Default | Description |
/// |-------|------|---------|-------------|
/// | `capacity` | `usize` | 1000 | Maximum number of entries |
/// | `enable_stats` | `bool` | false | Enable hit/miss tracking |
/// | `prealloc_memory` | `bool` | true | Pre-allocate memory for capacity |
/// | `thread_safe` | `bool` | false | Use internal synchronization |
///
/// # Example
///
/// ```
/// use cachekit::traits::CacheConfig;
///
/// // Use defaults
/// let config = CacheConfig::default();
/// assert_eq!(config.capacity, 1000);
/// assert!(!config.enable_stats);
///
/// // Custom configuration
/// let config = CacheConfig {
///     capacity: 5000,
///     enable_stats: true,
///     ..Default::default()
/// };
/// assert_eq!(config.capacity, 5000);
/// assert!(config.enable_stats);
/// assert!(config.prealloc_memory);  // from default
/// ```
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of entries the cache can hold.
    pub capacity: usize,

    /// Enable hit/miss statistics tracking.
    ///
    /// When enabled, the cache tracks hit rate, miss rate, and other metrics.
    /// Has a small performance overhead.
    pub enable_stats: bool,

    /// Pre-allocate memory for the full capacity.
    ///
    /// When true, memory is allocated upfront to avoid reallocations.
    /// When false, memory grows as needed (may cause latency spikes).
    pub prealloc_memory: bool,

    /// Use internal synchronization for thread safety.
    ///
    /// When true, the cache uses internal locks for thread-safe operations.
    /// When false, external synchronization (e.g., `Arc<RwLock<Cache>>`) is required.
    pub thread_safe: bool,
}

impl Default for CacheConfig {
    /// Creates a default configuration.
    ///
    /// Defaults:
    /// - `capacity`: 1000
    /// - `enable_stats`: false
    /// - `prealloc_memory`: true
    /// - `thread_safe`: false
    fn default() -> Self {
        Self {
            capacity: 1000,
            enable_stats: false,
            prealloc_memory: true,
            thread_safe: false,
        }
    }
}

/// Extension trait for async cache operations.
///
/// This trait is a placeholder for future async cache support. It will be
/// fully implemented in Phase 2 when the `async-trait` dependency is added.
///
/// Currently, all methods return `false` indicating async operations are
/// not supported. Implementations can override these to indicate support.
///
/// # Future API (Phase 2)
///
/// ```ignore
/// // Future async methods (not yet implemented)
/// async fn async_get(&self, key: &K) -> Option<&V>;
/// async fn async_insert(&mut self, key: K, value: V) -> Option<V>;
/// ```
///
/// # Example
///
/// ```
/// use cachekit::traits::AsyncCacheFuture;
///
/// struct MyCache;
///
/// impl AsyncCacheFuture<u64, String> for MyCache {
///     // Use defaults (no async support)
/// }
///
/// let cache = MyCache;
/// assert!(!cache.supports_async_get());
/// assert!(!cache.supports_async_insert());
/// ```
pub trait AsyncCacheFuture<K, V>: Send + Sync {
    /// Returns whether this cache supports async get operations.
    ///
    /// Default returns `false`. Override to indicate async support.
    fn supports_async_get(&self) -> bool {
        false
    }

    /// Returns whether this cache supports async insert operations.
    ///
    /// Default returns `false`. Override to indicate async support.
    fn supports_async_insert(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock implementation for testing trait design
    struct MockFifoCache {
        data: Vec<(i32, String)>,
        capacity: usize,
    }

    impl CoreCache<i32, String> for MockFifoCache {
        fn insert(&mut self, key: i32, value: String) -> Option<String> {
            // Simple mock implementation
            if let Some((_, existing)) = self.data.iter_mut().find(|(k, _)| *k == key) {
                return Some(std::mem::replace(existing, value));
            }
            if self.data.len() >= self.capacity {
                self.data.remove(0);
            }
            self.data.push((key, value));
            None
        }

        fn get(&mut self, key: &i32) -> Option<&String> {
            self.data.iter().find(|(k, _)| k == key).map(|(_, v)| v)
        }

        fn contains(&self, key: &i32) -> bool {
            self.data.iter().any(|(k, _)| k == key)
        }

        fn len(&self) -> usize {
            self.data.len()
        }

        fn capacity(&self) -> usize {
            self.capacity
        }

        fn clear(&mut self) {
            self.data.clear();
        }
    }

    impl FifoCacheTrait<i32, String> for MockFifoCache {
        fn pop_oldest(&mut self) -> Option<(i32, String)> {
            if self.data.is_empty() {
                None
            } else {
                Some(self.data.remove(0))
            }
        }

        fn peek_oldest(&self) -> Option<(&i32, &String)> {
            self.data.first().map(|(k, v)| (k, v))
        }

        fn age_rank(&self, key: &i32) -> Option<usize> {
            self.data.iter().position(|(k, _)| k == key)
        }
    }

    #[test]
    fn test_fifo_trait_design() {
        let mut cache = MockFifoCache {
            data: Vec::new(),
            capacity: 2,
        };

        // Test CoreCache operations
        cache.insert(1, "first".to_string());
        cache.insert(2, "second".to_string());
        assert_eq!(cache.len(), 2);
        assert!(cache.contains(&1));

        // Test FIFO operations
        assert_eq!(cache.peek_oldest(), Some((&1, &"first".to_string())));
        assert_eq!(cache.pop_oldest(), Some((1, "first".to_string())));
        assert_eq!(cache.len(), 1);

        // Test that FIFO cache doesn't have remove method
        // This won't compile - which is exactly what we want!
        // cache.remove(&2); // ❌ Compile error - good!
    }

    #[test]
    fn test_cache_config() {
        let config = CacheConfig {
            capacity: 500,
            enable_stats: true,
            ..Default::default()
        };

        assert_eq!(config.capacity, 500);
        assert!(config.enable_stats);
        assert!(config.prealloc_memory); // from default
    }

    #[test]
    fn test_core_cache_insert_returns_previous_value() {
        let mut cache = MockFifoCache {
            data: Vec::new(),
            capacity: 2,
        };

        assert_eq!(cache.insert(1, "first".to_string()), None);
        assert_eq!(
            cache.insert(1, "second".to_string()),
            Some("first".to_string())
        );
        assert_eq!(cache.get(&1), Some(&"second".to_string()));
    }
}
