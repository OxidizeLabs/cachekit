//! # Cache Trait Hierarchy
//!
//! This module defines the trait hierarchy for the cache subsystem, providing a unified
//! interface for different cache eviction policies (FIFO, LRU, LFU, LRU-K) while ensuring
//! type safety and policy-appropriate operation sets.
//!
//! ## Architecture
//!
//! ```text
//!                             ┌─────────────────────────────────────────┐
//!                             │            CoreCache<K, V>              │
//!                             │                                         │
//!                             │  insert(&mut, K, V) → Option<V>         │
//!                             │  get(&mut, &K) → Option<&V>             │
//!                             │  contains(&, &K) → bool                 │
//!                             │  len(&) → usize                         │
//!                             │  is_empty(&) → bool                     │
//!                             │  capacity(&) → usize                    │
//!                             │  clear(&mut)                            │
//!                             └──────────────────┬──────────────────────┘
//!                                                │
//!                    ┌───────────────────────────┼───────────────────────────┐
//!                    │                           │                           │
//!                    ▼                           ▼                           ▼
//!   ┌────────────────────────────┐   ┌─────────────────────────┐   ┌───────────────────────┐
//!   │   FIFOCacheTrait<K, V>     │   │   MutableCache<K, V>    │   │  DatabaseCache<K, V>  │
//!   │                            │   │                         │   │                       │
//!   │  pop_oldest() → (K, V)     │   │  remove(&K) → Option<V> │   │  evict_batch(n)       │
//!   │  peek_oldest() → (&K, &V)  │   │  remove_batch(&[K])     │   │  is_under_pressure()  │
//!   │  pop_oldest_batch(n)       │   │                         │   │  memory_usage_bytes() │
//!   │  age_rank(&K) → usize      │   └───────────┬─────────────┘   │  prepare_shutdown()   │
//!   │                            │               │                 └───────────────────────┘
//!   │  No arbitrary removal!     │               │
//!   └────────────────────────────┘               │
//!                                    ┌───────────┴───────────┐
//!                                    │                       │
//!                                    ▼                       ▼
//!                     ┌──────────────────────────┐  ┌────────────────────────┐
//!                     │   LRUCacheTrait<K, V>    │  │   LFUCacheTrait<K, V>  │
//!                     │                          │  │                        │
//!                     │  pop_lru() → (K, V)      │  │  pop_lfu() → (K, V)    │
//!                     │  peek_lru() → (&K, &V)   │  │  peek_lfu() → (&K, &V) │
//!                     │  touch(&K) → bool        │  │  frequency(&K) → u64   │
//!                     │  recency_rank(&K)        │  │  reset_frequency(&K)   │
//!                     │                          │  │  increment_frequency() │
//!                     └──────────────────────────┘  └────────────────────────┘
//!                                    │
//!                                    ▼
//!                     ┌──────────────────────────┐
//!                     │  LRUKCacheTrait<K, V>    │
//!                     │                          │
//!                     │  pop_lru_k() → (K, V)    │
//!                     │  peek_lru_k() → (&K,&V)  │
//!                     │  k_value() → usize       │
//!                     │  access_history(&K)      │
//!                     │  access_count(&K)        │
//!                     │  k_distance(&K) → u64    │
//!                     │  touch(&K) → bool        │
//!                     │  k_distance_rank(&K)     │
//!                     └──────────────────────────┘
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
//! | `FIFOCacheTrait`     | `CoreCache`     | FIFO-specific (no remove!)           |
//! | `LRUCacheTrait`      | `MutableCache`  | LRU-specific with recency tracking   |
//! | `LFUCacheTrait`      | `MutableCache`  | LFU-specific with frequency tracking |
//! | `LRUKCacheTrait`     | `MutableCache`  | LRU-K with K-distance tracking       |
//! | `ConcurrentCache`    | `Send + Sync`   | Marker for thread-safe caches        |
//! | `CacheStats`         | -               | Hit ratio and monitoring             |
//! | `DatabaseCache`      | `CoreCache`     | Database-specific operations         |
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
//!   Solution: FIFOCacheTrait extends CoreCache directly, ensuring
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
//!
//!   ┌─────────────────────────────────────────────────────────────────────────┐
//!   │ CacheStats                                                              │
//!   │                                                                         │
//!   │   hit_ratio()      → f64   (0.0 to 1.0)                                 │
//!   │   total_gets()     → u64                                                │
//!   │   total_hits()     → u64                                                │
//!   │   total_misses()   → u64   (default: gets - hits)                       │
//!   │   total_evictions()→ u64                                                │
//!   │   reset_stats()                                                         │
//!   └─────────────────────────────────────────────────────────────────────────┘
//!
//!   ┌─────────────────────────────────────────────────────────────────────────┐
//!   │ DatabaseCache                                                           │
//!   │                                                                         │
//!   │   evict_batch(target_size) → Vec<(K, V)>   Bulk eviction                │
//!   │   is_under_pressure()      → bool          len/capacity > 0.9           │
//!   │   memory_usage_bytes()     → usize         Approximate memory usage     │
//!   │   prepare_shutdown()                       Flush pending operations     │
//!   └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Cache Tier Management
//!
//! ```text
//!   ┌─────────────────────────────────────────────────────────────────────────┐
//!   │                         Three-Tier Cache Architecture                   │
//!   │                                                                         │
//!   │   ┌──────────────┐    promote()    ┌──────────────┐    promote()        │
//!   │   │  Cold Tier   │ ───────────────►│  Warm Tier   │───────────────►     │
//!   │   │              │                 │              │                     │
//!   │   │  FIFOCache   │◄─────────────── │  LFUCache    │◄───────────────     │
//!   │   │              │    demote()     │              │    demote()         │
//!   │   └──────────────┘                 └──────────────┘                     │
//!   │         │                                │                              │
//!   │         │                                │          ┌──────────────┐    │
//!   │         │                                │          │   Hot Tier   │    │
//!   │         │                                └─────────►│              │    │
//!   │         │                                           │  LRUCache    │    │
//!   │         └──────────────────────────────────────────►│              │    │
//!   │                                                     └──────────────┘    │
//!   │                                                                         │
//!   │   locate_key(&K)       → CacheTier   Which tier has this key?           │
//!   │   evict_from_tier(T)   → (K, V)      Force eviction from tier           │
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
//!     CoreCache, MutableCache, FIFOCacheTrait, LRUCacheTrait, LFUCacheTrait,
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
//! fn evict_oldest_batch<C: FIFOCacheTrait<u64, Vec<u8>>>(
//!     cache: &mut C,
//!     count: usize,
//! ) -> Vec<(u64, Vec<u8>)> {
//!     cache.pop_oldest_batch(count)
//! }
//!
//! // LRU-specific function
//! fn touch_hot_keys<C: LRUCacheTrait<u64, Vec<u8>>>(cache: &mut C, keys: &[u64]) {
//!     for key in keys {
//!         cache.touch(key); // Mark as recently used without retrieving
//!     }
//! }
//!
//! // LFU-specific function with frequency-based prioritization
//! fn boost_key_priority<C: LFUCacheTrait<u64, Vec<u8>>>(cache: &mut C, key: &u64) {
//!     // Increment frequency without accessing value
//!     cache.increment_frequency(key);
//! }
//!
//! // Thread-safe cache usage
//! use std::sync::{Arc, RwLock};
//! use crate::storage::disk::async_disk::cache::lru::ConcurrentLRUCache;
//!
//! let shared_cache = Arc::new(ConcurrentLRUCache::<u64, Vec<u8>>::new(1000));
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
//! - Some implementations (e.g., `ConcurrentLRUCache`) provide built-in concurrency
//!
//! ## Implementation Notes
//!
//! - **Trait Bounds**: `CoreCache` has no bounds on K, V; implementations add as needed
//! - **Default Implementations**: `is_empty()`, `total_misses()`, `is_under_pressure()`
//! - **Batch Operations**: Default implementations loop over single operations
//! - **Async Support**: `AsyncCacheFuture` prepared for Phase 2 async-trait integration

use std::hash::Hash;

/// Core cache operations that all caches support.
///
/// This trait focuses on the essential operations that make sense for any cache type,
/// regardless of eviction policy. All policy-specific traits extend this.
pub trait CoreCache<K, V> {
    /// Insert a key-value pair, returning the previous value if it existed
    fn insert(&mut self, key: K, value: V) -> Option<V>;

    /// Get a value by key (may update internal state for access tracking)
    fn get(&mut self, key: &K) -> Option<&V>;

    /// Check if a key exists without potentially updating access state
    fn contains(&self, key: &K) -> bool;

    /// Get the current number of entries
    fn len(&self) -> usize;

    /// Check if the cache is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the maximum capacity
    fn capacity(&self) -> usize;

    /// Remove all entries
    fn clear(&mut self);
}

/// Caches that support arbitrary key-based removal
/// This is appropriate for LRU, LFU, and general hash-map style caches
pub trait MutableCache<K, V>: CoreCache<K, V> {
    /// Remove a specific key-value pair
    /// Returns the removed value if the key existed
    fn remove(&mut self, key: &K) -> Option<V>;

    /// Remove multiple keys efficiently
    fn remove_batch(&mut self, keys: &[K]) -> Vec<Option<V>> {
        keys.iter().map(|k| self.remove(k)).collect()
    }
}

/// FIFO-specific operations that respect insertion order
/// No arbitrary removal - only ordered operations that maintain FIFO semantics
pub trait FIFOCacheTrait<K, V>: CoreCache<K, V> {
    /// Remove and return the oldest entry (first inserted)
    fn pop_oldest(&mut self) -> Option<(K, V)>;

    /// Peek at the oldest entry without removing it
    fn peek_oldest(&self) -> Option<(&K, &V)>;

    /// Remove multiple oldest entries efficiently
    fn pop_oldest_batch(&mut self, count: usize) -> Vec<(K, V)> {
        (0..count).filter_map(|_| self.pop_oldest()).collect()
    }

    /// Get the age rank of a key (0 = oldest, higher = newer)
    /// Returns None if key not found
    fn age_rank(&self, key: &K) -> Option<usize>;
}

/// LRU-specific operations that respect access order
pub trait LRUCacheTrait<K, V>: MutableCache<K, V> {
    /// Remove and return the least recently used entry
    fn pop_lru(&mut self) -> Option<(K, V)>;

    /// Peek at the LRU entry without removing it
    fn peek_lru(&self) -> Option<(&K, &V)>;

    /// Touch an entry to mark it as recently used without retrieving the value
    /// Returns true if the key was found and touched
    fn touch(&mut self, key: &K) -> bool;

    /// Get the recency rank of a key (0 = most recent, higher = less recent)
    /// Returns None if key not found
    fn recency_rank(&self, key: &K) -> Option<usize>;
}

/// LFU-specific operations that respect frequency order
pub trait LFUCacheTrait<K, V>: MutableCache<K, V> {
    /// Remove and return the least frequently used entry
    fn pop_lfu(&mut self) -> Option<(K, V)>;

    /// Peek at the LFU entry without removing it
    fn peek_lfu(&self) -> Option<(&K, &V)>;

    /// Get the access frequency for a key
    fn frequency(&self, key: &K) -> Option<u64>;

    /// Reset the frequency counter for a key to 1
    /// Returns the old frequency if the key existed
    fn reset_frequency(&mut self, key: &K) -> Option<u64>;

    /// Increment frequency without accessing the value
    /// Returns the new frequency if the key existed
    fn increment_frequency(&mut self, key: &K) -> Option<u64>;
}

/// LRU-K specific operations that respect K-distance access patterns
pub trait LRUKCacheTrait<K, V>: MutableCache<K, V> {
    /// Remove and return the entry with the oldest K-th access time
    fn pop_lru_k(&mut self) -> Option<(K, V)>;

    /// Peek at the LRU-K entry without removing it
    fn peek_lru_k(&self) -> Option<(&K, &V)>;

    /// Get the K value used by this cache
    fn k_value(&self) -> usize;

    /// Get the access history for a key (most recent first)
    /// Returns None if key not found
    fn access_history(&self, key: &K) -> Option<Vec<u64>>;

    /// Get the number of recorded accesses for a key
    fn access_count(&self, key: &K) -> Option<usize>;

    /// Get the K-th most recent access time for a key (if it has K accesses)
    /// Returns None if key not found or has fewer than K accesses
    fn k_distance(&self, key: &K) -> Option<u64>;

    /// Touch an entry to record an access without retrieving the value
    /// Returns true if the key was found and touched
    fn touch(&mut self, key: &K) -> bool;

    /// Get the rank of a key based on K-distance (0 = oldest K-distance, higher = newer K-distance)
    /// Entries with fewer than K accesses are ranked by their earliest access time
    /// Returns None if key not found
    fn k_distance_rank(&self, key: &K) -> Option<usize>;
}

/// Marker trait for caches that are safe to use concurrently
/// Implementors guarantee thread-safe operations
pub trait ConcurrentCache: Send + Sync {}

/// Statistics and monitoring capabilities
pub trait CacheStats {
    /// Cache hit ratio (0.0 to 1.0)
    fn hit_ratio(&self) -> f64;

    /// Total number of get operations
    fn total_gets(&self) -> u64;

    /// Total number of cache hits
    fn total_hits(&self) -> u64;

    /// Total number of cache misses
    fn total_misses(&self) -> u64 {
        self.total_gets() - self.total_hits()
    }

    /// Total number of evictions
    fn total_evictions(&self) -> u64;

    /// Reset all statistics
    fn reset_stats(&mut self);
}

/// Database-specific cache operations
pub trait DatabaseCache<K, V>: CoreCache<K, V> {
    /// Bulk eviction for memory pressure scenarios
    fn evict_batch(&mut self, target_size: usize) -> Vec<(K, V)>;

    /// Check if the cache is under memory pressure
    fn is_under_pressure(&self) -> bool {
        self.len() as f64 / self.capacity() as f64 > 0.9
    }

    /// Get memory usage in bytes (approximate)
    fn memory_usage_bytes(&self) -> usize;

    /// Prepare for shutdown (flush pending operations, etc.)
    fn prepare_shutdown(&mut self);
}

/// High-level cache tier management
pub trait CacheTierManager<K, V> {
    type HotCache: LRUCacheTrait<K, V> + ConcurrentCache;
    type WarmCache: LFUCacheTrait<K, V> + ConcurrentCache;
    type ColdCache: FIFOCacheTrait<K, V> + ConcurrentCache;

    /// Promote an entry from a lower tier to a higher tier
    fn promote(&mut self, key: &K, from_tier: CacheTier, to_tier: CacheTier) -> bool;

    /// Demote an entry from a higher tier to a lower tier
    fn demote(&mut self, key: &K, from_tier: CacheTier, to_tier: CacheTier) -> bool;

    /// Get the tier where a key currently resides
    fn locate_key(&self, key: &K) -> Option<CacheTier>;

    /// Force eviction from a specific tier
    fn evict_from_tier(&mut self, tier: CacheTier) -> Option<(K, V)>;
}

/// Cache tier enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CacheTier {
    /// Hot tier: frequently accessed data (LRU-managed)
    Hot,
    /// Warm tier: moderately accessed data (LFU-managed)
    Warm,
    /// Cold tier: rarely accessed data (FIFO-managed)
    Cold,
}

/// Factory trait for creating cache instances
pub trait CacheFactory<K, V> {
    type Cache: CoreCache<K, V>;

    /// Create a new cache instance with the specified capacity
    fn create(capacity: usize) -> Self::Cache;

    /// Create a cache with custom configuration
    fn create_with_config(config: CacheConfig) -> Self::Cache;
}

/// Configuration for cache creation
#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub capacity: usize,
    pub enable_stats: bool,
    pub prealloc_memory: bool,
    pub thread_safe: bool,
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

/// Extension trait for async cache operations
/// Note: Will be implemented in Phase 2 when async-trait dependency is added
pub trait AsyncCacheFuture<K, V>: Send + Sync {
    /// Placeholder for future async get operation
    fn supports_async_get(&self) -> bool {
        false
    }

    /// Placeholder for future async insert operation
    fn supports_async_insert(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock implementation for testing trait design
    struct MockFIFOCache {
        data: Vec<(i32, String)>,
        capacity: usize,
    }

    impl CoreCache<i32, String> for MockFIFOCache {
        fn insert(&mut self, key: i32, value: String) -> Option<String> {
            // Simple mock implementation
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

    impl FIFOCacheTrait<i32, String> for MockFIFOCache {
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
        let mut cache = MockFIFOCache {
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
}
