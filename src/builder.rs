//! Unified cache builder for all eviction policies.
//!
//! Provides a simple API to create caches with different eviction policies
//! while hiding the internal implementation details (like `Arc<V>` wrapping).
//!
//! ## Example
//!
//! ```rust
//! use cachekit::builder::{CacheBuilder, CachePolicy};
//!
//! let mut cache = CacheBuilder::new(100).build::<u64, String>(CachePolicy::Lru);
//! cache.insert(1, "hello".to_string());
//! assert_eq!(cache.get(&1), Some(&"hello".to_string()));
//! ```

use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;

use crate::policy::fifo::FifoCache;
use crate::policy::heap_lfu::HeapLfuCache;
use crate::policy::lfu::LfuCache;
use crate::policy::lru::LruCore;
use crate::policy::lru_k::LrukCache;
use crate::policy::two_q::TwoQCore;
use crate::traits::CoreCache;

/// Available cache eviction policies.
#[derive(Debug, Clone)]
pub enum CachePolicy {
    /// First In, First Out eviction.
    Fifo,
    /// Least Recently Used eviction.
    Lru,
    /// LRU-K policy with configurable K value (number of accesses to track).
    LruK { k: usize },
    /// Least Frequently Used eviction (bucket-based).
    Lfu,
    /// Least Frequently Used eviction (heap-based, requires `K: Ord`).
    HeapLfu,
    /// 2Q policy with configurable probation fraction.
    TwoQ { probation_frac: f64 },
}

/// Unified cache wrapper that provides a consistent API regardless of policy.
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
    Fifo(FifoCache<K, V>),
    Lru(LruCore<K, V>),
    LruK(LrukCache<K, V>),
    Lfu(LfuCache<K, V>),
    HeapLfu(HeapLfuCache<K, V>),
    TwoQ(TwoQCore<K, V>),
}

impl<K, V> Cache<K, V>
where
    K: Copy + Eq + Hash + Ord,
    V: Clone + Debug,
{
    /// Insert a key-value pair. Returns the previous value if the key existed.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        match &mut self.inner {
            CacheInner::Fifo(fifo) => CoreCache::insert(fifo, key, value),
            CacheInner::Lru(lru) => {
                let arc_value = Arc::new(value);
                lru.insert(key, arc_value)
                    .map(|arc| Arc::try_unwrap(arc).unwrap_or_else(|arc| (*arc).clone()))
            },
            CacheInner::LruK(lruk) => CoreCache::insert(lruk, key, value),
            CacheInner::Lfu(lfu) => {
                let arc_value = Arc::new(value);
                lfu.insert(key, arc_value)
                    .map(|arc| Arc::try_unwrap(arc).unwrap_or_else(|arc| (*arc).clone()))
            },
            CacheInner::HeapLfu(heap_lfu) => {
                let arc_value = Arc::new(value);
                heap_lfu
                    .insert(key, arc_value)
                    .map(|arc| Arc::try_unwrap(arc).unwrap_or_else(|arc| (*arc).clone()))
            },
            CacheInner::TwoQ(twoq) => CoreCache::insert(twoq, key, value),
        }
    }

    /// Get a reference to a value by key.
    pub fn get(&mut self, key: &K) -> Option<&V> {
        match &mut self.inner {
            CacheInner::Fifo(fifo) => fifo.get(key),
            CacheInner::Lru(lru) => lru.get(key).map(|arc| arc.as_ref()),
            CacheInner::LruK(lruk) => lruk.get(key),
            CacheInner::Lfu(lfu) => lfu.get(key).map(|arc| arc.as_ref()),
            CacheInner::HeapLfu(heap_lfu) => heap_lfu.get(key).map(|arc| arc.as_ref()),
            CacheInner::TwoQ(twoq) => twoq.get(key),
        }
    }

    /// Check if a key exists.
    pub fn contains(&self, key: &K) -> bool {
        match &self.inner {
            CacheInner::Fifo(fifo) => fifo.contains(key),
            CacheInner::Lru(lru) => lru.contains(key),
            CacheInner::LruK(lruk) => lruk.contains(key),
            CacheInner::Lfu(lfu) => lfu.contains(key),
            CacheInner::HeapLfu(heap_lfu) => heap_lfu.contains(key),
            CacheInner::TwoQ(twoq) => twoq.contains(key),
        }
    }

    /// Return the number of entries.
    pub fn len(&self) -> usize {
        match &self.inner {
            CacheInner::Fifo(fifo) => CoreCache::len(fifo),
            CacheInner::Lru(lru) => lru.len(),
            CacheInner::LruK(lruk) => CoreCache::len(lruk),
            CacheInner::Lfu(lfu) => lfu.len(),
            CacheInner::HeapLfu(heap_lfu) => heap_lfu.len(),
            CacheInner::TwoQ(twoq) => twoq.len(),
        }
    }

    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the maximum capacity.
    pub fn capacity(&self) -> usize {
        match &self.inner {
            CacheInner::Fifo(fifo) => CoreCache::capacity(fifo),
            CacheInner::Lru(lru) => lru.capacity(),
            CacheInner::LruK(lruk) => CoreCache::capacity(lruk),
            CacheInner::Lfu(lfu) => lfu.capacity(),
            CacheInner::HeapLfu(heap_lfu) => heap_lfu.capacity(),
            CacheInner::TwoQ(twoq) => twoq.capacity(),
        }
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        match &mut self.inner {
            CacheInner::Fifo(fifo) => fifo.clear(),
            CacheInner::Lru(lru) => lru.clear(),
            CacheInner::LruK(lruk) => lruk.clear(),
            CacheInner::Lfu(lfu) => lfu.clear(),
            CacheInner::HeapLfu(heap_lfu) => heap_lfu.clear(),
            CacheInner::TwoQ(twoq) => twoq.clear(),
        }
    }
}

/// Builder for creating cache instances.
pub struct CacheBuilder {
    capacity: usize,
}

impl CacheBuilder {
    /// Create a new cache builder with the specified capacity.
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
            CachePolicy::Fifo => CacheInner::Fifo(FifoCache::new(self.capacity)),
            CachePolicy::Lru => CacheInner::Lru(LruCore::new(self.capacity)),
            CachePolicy::LruK { k } => CacheInner::LruK(LrukCache::with_k(self.capacity, k)),
            CachePolicy::Lfu => CacheInner::Lfu(LfuCache::new(self.capacity)),
            CachePolicy::HeapLfu => CacheInner::HeapLfu(HeapLfuCache::new(self.capacity)),
            CachePolicy::TwoQ { probation_frac } => {
                CacheInner::TwoQ(TwoQCore::new(self.capacity, probation_frac))
            },
        };

        Cache { inner }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_policies_basic_ops() {
        let policies = [
            CachePolicy::Fifo,
            CachePolicy::Lru,
            CachePolicy::LruK { k: 2 },
            CachePolicy::Lfu,
            CachePolicy::HeapLfu,
            CachePolicy::TwoQ {
                probation_frac: 0.25,
            },
        ];

        for policy in policies {
            let mut cache = CacheBuilder::new(10).build::<u64, String>(policy.clone());

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
