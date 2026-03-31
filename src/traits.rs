//! # Cache Trait Hierarchy
//!
//! This module defines a pluggable-policy cache interface. Users code against the
//! main [`Cache`] trait to swap eviction policies with minimal code changes;
//! optional capability traits expose advanced behaviour for users who need it.
//!
//! ## Architecture
//!
//! ```text
//!   ┌────────────────────────────────────────────────────────────────────┐
//!   │                      Cache<K, V>                                  │
//!   │  Main trait — every policy implements this.                       │
//!   │                                                                   │
//!   │  contains, len, is_empty, capacity                                │
//!   │  peek (side-effect-free), get (policy-tracked)                    │
//!   │  insert, remove, clear                                            │
//!   └────────────────────┬──────────────────────────────────────────────┘
//!                        │
//!     ┌──────────────────┼──────────────────┬───────────────────────┐
//!     │                  │                  │                       │
//!     ▼                  ▼                  ▼                       ▼
//!   EvictingCache   VictimInspectable   RecencyTracking    FrequencyTracking
//!   evict_one()     peek_victim()       touch, rank        frequency
//!                                                                  │
//!                                                          HistoryTracking
//!                                                          access_history,
//!                                                          k_distance, ...
//! ```
//!
//! ## Trait Summary
//!
//! | Trait                  | Extends     | Purpose                                       |
//! |------------------------|-------------|-----------------------------------------------|
//! | [`Cache`]              | —           | Universal cache operations (pluggable)        |
//! | [`EvictingCache`]      | `Cache`     | Explicit policy-driven eviction               |
//! | [`VictimInspectable`]  | `Cache`     | Read-only next-victim peek                    |
//! | [`RecencyTracking`]    | `Cache`     | Touch and recency-rank inspection             |
//! | [`FrequencyTracking`]  | `Cache`     | Access-frequency inspection                   |
//! | [`HistoryTracking`]    | `Cache`     | LRU-K style access-history inspection         |
//! | [`ConcurrentCache`]    | `Send+Sync` | Thread-safety marker                          |
//! | [`CacheFactory`]       | —           | Cache instance creation                       |
//! | [`AsyncCacheFuture`]   | `Send+Sync` | Future async operation support                |
//!
//! ## Example Usage
//!
//! ```
//! use cachekit::traits::Cache;
//! use cachekit::policy::lru_k::LrukCache;
//!
//! // Generic function — works with any policy
//! fn warm_cache<C: Cache<u64, String>>(cache: &mut C, data: &[(u64, String)]) {
//!     for (key, value) in data {
//!         cache.insert(*key, value.clone());
//!     }
//! }
//!
//! let mut cache = LrukCache::new(100);
//! warm_cache(&mut cache, &[(1, "one".to_string()), (2, "two".to_string())]);
//! assert_eq!(cache.len(), 2);
//! assert_eq!(cache.peek(&1), Some(&"one".to_string()));
//! ```
//!
//! ## Thread Safety
//!
//! - Individual cache implementations are **NOT thread-safe** by default
//! - Use [`ConcurrentCache`] marker trait to identify thread-safe implementations
//! - Wrap non-concurrent caches in `Arc<RwLock<C>>` for shared access
//! - Some implementations (e.g., `ConcurrentLruCache`) provide built-in concurrency
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
//! ## Implementation Notes
//!
//! - **Object safety**: [`Cache`] is intentionally object-safe (`Box<dyn Cache<K, V>>`)
//! - **Default Implementations**: `is_empty()`
//! - **Batch Operations**: Stay as inherent methods for buffer-reuse ergonomics
//! - **Async Support**: [`AsyncCacheFuture`] prepared for Phase 2 async-trait integration

// ---------------------------------------------------------------------------
// Layer 1 — Main trait
// ---------------------------------------------------------------------------

/// Universal cache operations that all policies implement.
///
/// This is the primary user-facing trait. Code written against `Cache<K, V>` can
/// swap eviction policies without changing call sites.
///
/// The trait is intentionally **object-safe** to support `Box<dyn Cache<K, V>>`.
///
/// # Type Parameters
///
/// - `K`: Key type (implementations typically require `Eq + Hash`)
/// - `V`: Value type
///
/// # Example
///
/// ```
/// use cachekit::traits::Cache;
/// use cachekit::policy::lru_k::LrukCache;
///
/// fn use_any_cache<C: Cache<u64, String>>(cache: &mut C) {
///     cache.insert(1, "hello".to_string());
///     assert_eq!(cache.peek(&1), Some(&"hello".to_string()));
///     assert_eq!(cache.get(&1), Some(&"hello".to_string()));
///     assert_eq!(cache.remove(&1), Some("hello".to_string()));
/// }
///
/// let mut cache = LrukCache::new(100);
/// use_any_cache(&mut cache);
/// ```
pub trait Cache<K, V> {
    /// Checks if a key exists without updating access state.
    fn contains(&self, key: &K) -> bool;

    /// Returns the current number of entries.
    fn len(&self) -> usize;

    /// Returns `true` if the cache contains no entries.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the maximum capacity of the cache.
    fn capacity(&self) -> usize;

    /// Side-effect-free lookup by key.
    ///
    /// Does not update access patterns, eviction order, or any internal state.
    fn peek(&self, key: &K) -> Option<&V>;

    /// Policy-tracked lookup by key.
    ///
    /// May update internal state (access time, frequency, reference bits)
    /// depending on the eviction policy. Use [`peek`](Self::peek) if you
    /// need a read without side effects.
    fn get(&mut self, key: &K) -> Option<&V>;

    /// Inserts a key-value pair, returning the previous value if it existed.
    ///
    /// If the cache is at capacity, an entry may be evicted according to the
    /// cache's eviction policy before the new entry is inserted.
    fn insert(&mut self, key: K, value: V) -> Option<V>;

    /// Removes a specific key-value pair, returning the value if it existed.
    fn remove(&mut self, key: &K) -> Option<V>;

    /// Removes all entries from the cache.
    fn clear(&mut self);
}

// ---------------------------------------------------------------------------
// Layer 2 — Optional capability traits
// ---------------------------------------------------------------------------

/// Explicitly evict one entry according to the policy.
///
/// Not all policies expose this; some (ARC, CAR, NRU, Random, SLRU, 2Q) only
/// evict implicitly during [`Cache::insert`].
///
/// # Example
///
/// ```
/// use cachekit::traits::{Cache, EvictingCache};
/// use cachekit::policy::fifo::FifoCache;
///
/// let mut cache = FifoCache::new(10);
/// cache.insert(1, "first");
/// cache.insert(2, "second");
///
/// let evicted = cache.evict_one();
/// assert_eq!(evicted, Some((1, "first")));
/// ```
pub trait EvictingCache<K, V>: Cache<K, V> {
    /// Removes and returns one entry selected by the eviction policy.
    ///
    /// Returns `None` if the cache is empty.
    #[must_use]
    fn evict_one(&mut self) -> Option<(K, V)>;
}

/// Read-only peek at the next eviction candidate.
///
/// Only implemented by policies where the victim is cheap and stable to
/// identify without mutating internal state. Policies that require sweeps
/// or adaptive decisions do not implement this.
///
/// # Example
///
/// ```
/// use cachekit::traits::{Cache, VictimInspectable};
/// use cachekit::policy::fifo::FifoCache;
///
/// let mut cache = FifoCache::new(10);
/// cache.insert(1, "first");
/// cache.insert(2, "second");
///
/// assert_eq!(cache.peek_victim(), Some((&1, &"first")));
/// assert_eq!(cache.len(), 2); // not removed
/// ```
pub trait VictimInspectable<K, V>: Cache<K, V> {
    /// Peeks at the entry that would be evicted next.
    ///
    /// Does not remove the entry or modify any state.
    fn peek_victim(&self) -> Option<(&K, &V)>;
}

/// Recency-based inspection and manipulation.
///
/// For policies that order entries by access recency (LRU, LRU-K, FastLRU).
///
/// # Example
///
/// ```
/// use std::sync::Arc;
/// use cachekit::traits::{Cache, RecencyTracking};
/// use cachekit::policy::lru::LruCore;
///
/// let mut cache: LruCore<u64, &str> = LruCore::new(10);
/// cache.insert(1, Arc::new("first"));
/// cache.insert(2, Arc::new("second"));
///
/// assert!(cache.touch(&1));
/// assert_eq!(cache.recency_rank(&1), Some(0)); // most recent
/// ```
pub trait RecencyTracking<K, V>: Cache<K, V> {
    /// Marks a key as recently used without retrieving its value.
    ///
    /// Returns `true` if the key was found and touched.
    fn touch(&mut self, key: &K) -> bool;

    /// Returns the recency rank (0 = most recent, higher = less recent).
    ///
    /// Returns `None` if the key is not found.
    fn recency_rank(&self, key: &K) -> Option<usize>;
}

/// Frequency-based inspection.
///
/// For policies that track access frequency (LFU, HeapLFU, MFU, LRU-K).
///
/// # Example
///
/// ```
/// use std::sync::Arc;
/// use cachekit::traits::{Cache, FrequencyTracking};
/// use cachekit::policy::lfu::LfuCache;
///
/// let mut cache: LfuCache<u64, &str> = LfuCache::new(10);
/// cache.insert(1, Arc::new("value"));
/// cache.get(&1);
/// assert_eq!(cache.frequency(&1), Some(2)); // 1 insert + 1 get
/// ```
pub trait FrequencyTracking<K, V>: Cache<K, V> {
    /// Returns the access frequency for a key.
    ///
    /// Returns `None` if the key is not found.
    fn frequency(&self, key: &K) -> Option<u64>;
}

/// LRU-K style access-history inspection.
///
/// Only implemented by `LrukCache`.
///
/// # Example
///
/// ```
/// use cachekit::traits::{Cache, HistoryTracking};
/// use cachekit::policy::lru_k::LrukCache;
///
/// let mut cache = LrukCache::with_k(10, 2);
/// cache.insert(1, "value");
/// cache.get(&1);
///
/// assert_eq!(cache.access_count(&1), Some(2));
/// assert!(cache.k_distance(&1).is_some());
/// assert_eq!(cache.k_value(), 2);
/// ```
pub trait HistoryTracking<K, V>: Cache<K, V> {
    /// Returns the total access count for a key.
    fn access_count(&self, key: &K) -> Option<usize>;

    /// Returns the K-distance for a key (time since the K-th most recent access).
    ///
    /// Returns `None` if the key has fewer than K accesses.
    fn k_distance(&self, key: &K) -> Option<u64>;

    /// Returns the access history timestamps (most recent first), up to K entries.
    fn access_history(&self, key: &K) -> Option<Vec<u64>>;

    /// Returns the K parameter used by this cache.
    fn k_value(&self) -> usize;
}

// ---------------------------------------------------------------------------
// Utility traits
// ---------------------------------------------------------------------------

/// Marker trait for caches that are safe to use concurrently.
///
/// Implementors guarantee thread-safe operations. This trait extends
/// `Send + Sync` and can be used as a bound to require concurrent access.
///
/// # Safety
///
/// Implementing this trait asserts that the cache handles internal
/// synchronization correctly. An incorrect implementation may lead to
/// data races when the cache is shared across threads.
///
/// # Example
///
/// ```
/// use cachekit::traits::{Cache, ConcurrentCache};
///
/// fn use_from_threads<K, V, C>(cache: &C)
/// where
///     K: Send + Sync,
///     V: Send + Sync,
///     C: Cache<K, V> + ConcurrentCache,
/// {
///     // Safe to share between threads
/// }
/// ```
pub unsafe trait ConcurrentCache: Send + Sync {}

/// Factory trait for creating cache instances.
///
/// Provides a standard interface for cache construction, allowing generic
/// code to create cache instances without knowing the concrete type.
///
/// # Example
///
/// ```ignore
/// use cachekit::traits::{Cache, CacheFactory, CacheConfig};
///
/// struct LruFactory;
///
/// impl CacheFactory<u64, String> for LruFactory {
///     type Cache = LruCache<u64, String>;
///
///     fn new(capacity: usize) -> Self::Cache {
///         LruCache::new(capacity)
///     }
///
///     fn with_config(config: CacheConfig) -> Self::Cache {
///         LruCache::new(config.capacity)
///     }
/// }
/// ```
pub trait CacheFactory<K, V> {
    /// The concrete cache type produced by this factory.
    type Cache: Cache<K, V>;

    /// Creates a new cache instance with the specified capacity.
    fn new(capacity: usize) -> Self::Cache;

    /// Creates a cache with custom configuration.
    fn with_config(config: CacheConfig) -> Self::Cache;
}

/// Configuration for cache creation.
///
/// Used with [`CacheFactory::with_config`] to customize cache behavior.
///
/// # Example
///
/// ```
/// use cachekit::traits::CacheConfig;
///
/// let config = CacheConfig::default();
/// assert_eq!(config.capacity, 1000);
/// assert!(!config.enable_stats);
///
/// let config = CacheConfig::new(5000).with_stats(true);
/// assert_eq!(config.capacity, 5000);
/// assert!(config.enable_stats);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct CacheConfig {
    /// Maximum number of entries the cache can hold.
    pub capacity: usize,

    /// Enable hit/miss statistics tracking.
    pub enable_stats: bool,

    /// Pre-allocate memory for the full capacity.
    pub prealloc_memory: bool,

    /// Use internal synchronization for thread safety.
    pub thread_safe: bool,
}

impl CacheConfig {
    /// Creates a new configuration with the given capacity and default options.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            ..Default::default()
        }
    }

    /// Enables or disables hit/miss statistics tracking.
    pub fn with_stats(mut self, enable: bool) -> Self {
        self.enable_stats = enable;
        self
    }

    /// Enables or disables upfront memory pre-allocation.
    pub fn with_prealloc(mut self, prealloc: bool) -> Self {
        self.prealloc_memory = prealloc;
        self
    }

    /// Enables or disables internal thread-safe synchronization.
    pub fn with_thread_safe(mut self, thread_safe: bool) -> Self {
        self.thread_safe = thread_safe;
        self
    }

    /// Validates the configuration, returning an error if any parameter is invalid.
    pub fn validate(&self) -> Result<(), crate::error::ConfigError> {
        if self.capacity == 0 {
            return Err(crate::error::ConfigError::new(
                "capacity must be greater than 0",
            ));
        }
        Ok(())
    }
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            capacity: 1000,
            enable_stats: false,
            prealloc_memory: true,
            thread_safe: false,
        }
    }
}

/// Extension trait for async cache operations (Phase 2 placeholder).
///
/// Currently all methods return `false`. Implementations can override
/// to indicate support.
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
    fn supports_async_get(&self) -> bool {
        false
    }

    /// Returns whether this cache supports async insert operations.
    fn supports_async_insert(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    struct MockCache {
        data: Vec<(i32, String)>,
        capacity: usize,
    }

    impl Cache<i32, String> for MockCache {
        fn contains(&self, key: &i32) -> bool {
            self.data.iter().any(|(k, _)| k == key)
        }

        fn len(&self) -> usize {
            self.data.len()
        }

        fn capacity(&self) -> usize {
            self.capacity
        }

        fn peek(&self, key: &i32) -> Option<&String> {
            self.data.iter().find(|(k, _)| k == key).map(|(_, v)| v)
        }

        fn get(&mut self, key: &i32) -> Option<&String> {
            self.data.iter().find(|(k, _)| k == key).map(|(_, v)| v)
        }

        fn insert(&mut self, key: i32, value: String) -> Option<String> {
            if let Some((_, existing)) = self.data.iter_mut().find(|(k, _)| *k == key) {
                return Some(std::mem::replace(existing, value));
            }
            if self.data.len() >= self.capacity {
                self.data.remove(0);
            }
            self.data.push((key, value));
            None
        }

        fn remove(&mut self, key: &i32) -> Option<String> {
            if let Some(pos) = self.data.iter().position(|(k, _)| k == key) {
                Some(self.data.remove(pos).1)
            } else {
                None
            }
        }

        fn clear(&mut self) {
            self.data.clear();
        }
    }

    impl EvictingCache<i32, String> for MockCache {
        fn evict_one(&mut self) -> Option<(i32, String)> {
            if self.data.is_empty() {
                None
            } else {
                Some(self.data.remove(0))
            }
        }
    }

    impl VictimInspectable<i32, String> for MockCache {
        fn peek_victim(&self) -> Option<(&i32, &String)> {
            self.data.first().map(|(k, v)| (k, v))
        }
    }

    #[test]
    fn test_cache_trait() {
        let mut cache = MockCache {
            data: Vec::new(),
            capacity: 2,
        };

        cache.insert(1, "first".to_string());
        cache.insert(2, "second".to_string());
        assert_eq!(cache.len(), 2);
        assert!(cache.contains(&1));
        assert_eq!(cache.peek(&1), Some(&"first".to_string()));
        assert_eq!(cache.get(&1), Some(&"first".to_string()));

        assert_eq!(cache.remove(&1), Some("first".to_string()));
        assert_eq!(cache.len(), 1);
        assert!(!cache.contains(&1));
    }

    #[test]
    fn test_evicting_cache() {
        let mut cache = MockCache {
            data: Vec::new(),
            capacity: 10,
        };

        cache.insert(1, "first".to_string());
        cache.insert(2, "second".to_string());

        assert_eq!(cache.peek_victim(), Some((&1, &"first".to_string())));
        assert_eq!(cache.evict_one(), Some((1, "first".to_string())));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_insert_returns_previous_value() {
        let mut cache = MockCache {
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

    #[test]
    fn test_cache_config() {
        let config = CacheConfig {
            capacity: 500,
            enable_stats: true,
            ..Default::default()
        };

        assert_eq!(config.capacity, 500);
        assert!(config.enable_stats);
        assert!(config.prealloc_memory);
    }

    #[test]
    fn test_object_safety() {
        let mut cache = MockCache {
            data: Vec::new(),
            capacity: 10,
        };
        cache.insert(1, "hello".to_string());

        let cache_ref: &dyn Cache<i32, String> = &cache;
        assert_eq!(cache_ref.len(), 1);
        assert_eq!(cache_ref.peek(&1), Some(&"hello".to_string()));
    }
}
