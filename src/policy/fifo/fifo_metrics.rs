//! # FIFO (First In, First Out) Cache Implementation
//!
//! This module provides a high-performance FIFO cache that evicts the oldest (first inserted)
//! items when capacity is reached. FIFO provides predictable, deterministic eviction behavior
//! without the complexity of tracking access patterns.
//!
//! ## Architecture
//!
//! ```text
//!   ┌──────────────────────────────────────────────────────────────────────────┐
//!   │                          FIFOCache<K, V>                                 │
//!   │                                                                          │
//!   │   ┌────────────────────────────────────────────────────────────────────┐ │
//!   │   │  cache: HashMap<K, Arc<V>>                                    │ │
//!   │   │                                                                    │ │
//!   │   │  ┌─────────────┬────────────────────────────────────────────────┐  │ │
//!   │   │  │   K    │  Arc<V>                                        │  │ │
//!   │   │  ├─────────────┼────────────────────────────────────────────────┤  │ │
//!   │   │  │  Arc(key1)  │  Arc(value1)                                   │  │ │
//!   │   │  │  Arc(key2)  │  Arc(value2)                                   │  │ │
//!   │   │  │  Arc(key3)  │  Arc(value3)                                   │  │ │
//!   │   │  └─────────────┴────────────────────────────────────────────────┘  │ │
//!   │   │                                                                    │ │
//!   │   │  O(1) lookup by key (via hash)                                     │ │
//!   │   └────────────────────────────────────────────────────────────────────┘ │
//!   │                                                                          │
//!   │   ┌────────────────────────────────────────────────────────────────────┐ │
//!   │   │  insertion_order: VecDeque<K>                                 │ │
//!   │   │                                                                    │ │
//!   │   │  front ──► [Arc(key1)] ─ [Arc(key2)] ─ [Arc(key3)] ◄── back        │ │
//!   │   │            (oldest)                      (newest)                  │ │
//!   │   │                                                                    │ │
//!   │   │  O(1) pop_front for FIFO eviction                                  │ │
//!   │   │  O(1) push_back for new insertions                                 │ │
//!   │   └────────────────────────────────────────────────────────────────────┘ │
//!   │                                                                          │
//!   │   capacity: usize                                                        │
//!   └──────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## FIFO Eviction Flow
//!
//! ```text
//!   insert(new_key, new_value)
//!        │
//!        ▼
//!   ┌────────────────────────────────────────────────────────────────────────┐
//!   │ Key already exists?                                                    │
//!   │                                                                        │
//!   │   YES → Update value, keep insertion position, return old value        │
//!   │   NO  → Continue to capacity check                                     │
//!   └────────────────────────────────────────────────────────────────────────┘
//!        │
//!        ▼
//!   ┌────────────────────────────────────────────────────────────────────────┐
//!   │ Cache at capacity?                                                     │
//!   │                                                                        │
//!   │   NO  → Add to HashMap + push_back to VecDeque                         │
//!   │   YES → Evict oldest, then add new entry                               │
//!   └────────────────────────────────────────────────────────────────────────┘
//!        │
//!        ▼ (capacity reached)
//!   ┌────────────────────────────────────────────────────────────────────────┐
//!   │ FIFO Eviction:                                                         │
//!   │                                                                        │
//!   │   1. pop_front() from VecDeque → oldest K                         │
//!   │   2. Skip if stale (key not in HashMap)                                │
//!   │   3. Remove from HashMap                                               │
//!   │   4. Add new entry: HashMap.insert + VecDeque.push_back                │
//!   │                                                                        │
//!   │   Eviction is O(1) amortized (may skip stale entries)                  │
//!   └────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## FIFO vs. Other Policies
//!
//! ```text
//!   Access pattern: A, B, C, A, A, A, D  (A accessed 4 times, others 1 each)
//!   Cache capacity: 3
//!
//!   FIFO (insertion-order based):
//!   ═══════════════════════════════════════════════════════════════════════════
//!     After A,B,C: [A, B, C]  (A = oldest, C = newest)
//!     After A,A,A: [A, B, C]  (accesses don't change order)
//!     Insert D:    [B, C, D]  ← A evicted (oldest) despite 4 accesses!
//!
//!   LRU (recency-based):
//!   ═══════════════════════════════════════════════════════════════════════════
//!     After A,B,C: [C, B, A]  (A = MRU)
//!     After A,A,A: [A, C, B]  (A moves to MRU each time)
//!     Insert D:    [D, A, C]  ← B evicted (LRU)
//!
//!   Key difference: FIFO ignores access patterns, LRU adapts to them.
//!   FIFO is simpler but can evict hot items.
//! ```
//!
//! ## Stale Entry Handling
//!
//! ```text
//!   Stale entries occur when:
//!   - An item is removed via remove() but VecDeque entry remains
//!   - The update path creates a new HashMap entry but keeps old VecDeque entry
//!
//!   ═══════════════════════════════════════════════════════════════════════════
//!
//!   Initial state:
//!     HashMap:     { A: v1, B: v2, C: v3 }
//!     VecDeque:    [A, B, C]
//!
//!   After remove(&B):
//!     HashMap:     { A: v1, C: v3 }         ← B removed
//!     VecDeque:    [A, B, C]                ← B still present (STALE)
//!
//!   During eviction:
//!     1. pop_front() → A (valid, evict)
//!     2. pop_front() → B (stale, skip)      ← Lazy cleanup
//!     3. pop_front() → C (valid, evict)
//!
//!   ═══════════════════════════════════════════════════════════════════════════
//! ```
//!
//! ## Key Components
//!
//! | Component         | Type                     | Purpose                       |
//! |-------------------|--------------------------|-------------------------------|
//! | `cache`           | `HashMap<K,Arc<V>>` | O(1) key-value storage        |
//! | `insertion_order` | `VecDeque<K>`       | Tracks insertion order        |
//! | `capacity`        | `usize`                  | Maximum entries               |
//!
//! ## Core Operations (CoreCache)
//!
//! | Method           | Complexity | Description                              |
//! |------------------|------------|------------------------------------------|
//! | `new(capacity)`  | O(1)       | Create cache with given capacity         |
//! | `insert(k, v)`   | O(1)*      | Insert/update, may trigger eviction      |
//! | `get(&k)`        | O(1)       | Get value (no order change in FIFO)      |
//! | `contains(&k)`   | O(1)       | Check if key exists                      |
//! | `len()`          | O(1)       | Current number of entries                |
//! | `capacity()`     | O(1)       | Maximum capacity                         |
//! | `clear()`        | O(n)       | Remove all entries                       |
//!
//! \* Amortized, may skip stale entries during eviction
//!
//! ## FIFO-Specific Operations (FIFOCacheTrait)
//!
//! | Method                | Complexity | Description                          |
//! |-----------------------|------------|--------------------------------------|
//! | `pop_oldest()`        | O(1)*      | Remove and return the oldest entry   |
//! | `peek_oldest()`       | O(n)*      | Peek at oldest without removing      |
//! | `pop_oldest_batch(n)` | O(n)       | Remove the n oldest entries          |
//! | `age_rank(&k)`        | O(n)       | Get position (0 = oldest)            |
//!
//! \* May need to skip stale entries
//!
//! ## Performance Characteristics
//!
//! | Operation          | Time       | Notes                              |
//! |--------------------|------------|------------------------------------|
//! | `get`              | O(1)       | HashMap lookup only                |
//! | `insert` (no evict)| O(1)       | HashMap + VecDeque push            |
//! | `insert` (evict)   | O(1)*      | + pop_front + HashMap remove       |
//! | `pop_oldest`       | O(1)*      | pop_front + HashMap remove         |
//! | `peek_oldest`      | O(n)*      | May scan for valid entry           |
//! | `age_rank`         | O(n)       | Linear scan of VecDeque            |
//! | Per-entry overhead | ~48 bytes  | 2 × Arc + HashMap + VecDeque entry |
//!
//! \* Amortized, skipping stale entries
//!
//! ## Trade-offs
//!
//! | Aspect           | Pros                              | Cons                            |
//! |------------------|-----------------------------------|---------------------------------|
//! | Simplicity       | No access tracking needed         | Can evict hot items             |
//! | Predictability   | Deterministic eviction order      | Ignores access patterns         |
//! | Performance      | O(1) get (no order update)        | O(n) for age_rank, peek         |
//! | Memory           | Arc enables zero-copy sharing     | Arc overhead per entry.         |
//!
//! ## When to Use
//!
//! **Use FIFO when:**
//! - Eviction order should be predictable and deterministic
//! - Access patterns don't strongly indicate item importance
//! - Simplicity is preferred over adaptive behavior
//! - All items have roughly equal access probability
//!
//! **Avoid FIFO when:**
//! - Hot items should be retained (use LRU or LFU)
//! - Access patterns indicate item importance
//! - Scan resistance is needed (use LRU-K)
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! use crate::storage::disk::async_disk::cache::fifo::FIFOCache;
//! use crate::storage::disk::async_disk::cache::cache_traits::{
//!     CoreCache, FIFOCacheTrait,
//! };
//!
//! // Create cache
//! let mut cache: FIFOCache<String, i32> = FIFOCache::new(100);
//!
//! // Insert items
//! cache.insert("key1".to_string(), 100);
//! cache.insert("key2".to_string(), 200);
//! cache.insert("key3".to_string(), 300);
//!
//! // Get doesn't affect order (unlike LRU)
//! if let Some(value) = cache.get(&"key1".to_string()) {
//!     println!("Got: {}", value);
//! }
//!
//! // Check age rank (0 = oldest)
//! assert_eq!(cache.age_rank(&"key1".to_string()), Some(0)); // oldest
//! assert_eq!(cache.age_rank(&"key3".to_string()), Some(2)); // newest
//!
//! // Peek at oldest without removing
//! if let Some((key, value)) = cache.peek_oldest() {
//!     println!("Oldest: {} = {}", key, value);
//! }
//!
//! // Pop oldest (FIFO eviction)
//! if let Some((key, value)) = cache.pop_oldest() {
//!     println!("Evicted: {} = {}", key, value);
//! }
//!
//! // Batch eviction
//! let evicted = cache.pop_oldest_batch(2);
//! println!("Batch evicted {} items", evicted.len());
//!
//! // Thread-safe usage
//! use std::sync::{Arc, RwLock};
//! let shared_cache = Arc::new(RwLock::new(FIFOCache::<u64, Vec<u8>>::new(1000)));
//!
//! // Write access
//! {
//!     let mut cache = shared_cache.write().unwrap();
//!     cache.insert(page_id, page_data);
//! }
//!
//! // Read access
//! {
//!     let cache = shared_cache.read().unwrap();
//!     if let Some(data) = cache.get(&page_id) {
//!         // use data
//!     }
//! }
//! ```
//!
//! ## Comparison with Other Policies
//!
//! | Policy   | Eviction Basis     | Get Time | Evict Time | Best For               |
//! |----------|--------------------|----------|------------|------------------------|
//! | FIFO     | Insertion order    | O(1)     | O(1)       | Predictable behavior   |
//! | LRU      | Recency            | O(1)*    | O(1)       | Temporal locality      |
//! | LFU      | Frequency          | O(1)     | O(n)       | Stable access patterns |
//! | LRU-K    | K-th access        | O(1)     | O(n)       | Scan resistance        |
//!
//! \* LRU get requires order update
//!
//! ## Thread Safety
//!
//! - `FIFOCache` is **NOT thread-safe**
//! - Use `ConcurrentFIFOCache` for thread-safe access
//! - Designed for single-threaded use with external synchronization
//! - Follows Rust's zero-cost abstractions principle
//!
//! ## Implementation Notes
//!
//! - **Arc Sharing**: Values are stored as `Arc<V>` inside the store for zero-copy sharing
//! - **Stale Entries**: VecDeque may contain entries not in the store (lazy cleanup)
//! - **Update Semantics**: Updating existing key preserves insertion position
//! - **Zero Capacity**: Supported - rejects all insertions

use crate::metrics::metrics_impl::CacheMetrics;
use crate::metrics::snapshot::CacheMetricsSnapshot;
use crate::metrics::traits::{
    CoreMetricsRecorder, FifoMetricsReadRecorder, FifoMetricsRecorder, MetricsSnapshotProvider,
};
use crate::store::hashmap::HashMapStore;
use crate::store::traits::{StoreCore, StoreMut};
use crate::traits::{ConcurrentCache, CoreCache, FIFOCacheTrait};
use parking_lot::RwLock;
use std::collections::VecDeque;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;

/// FIFO (First In, First Out) Cache.
///
/// Evicts the oldest (first inserted) item when capacity is reached.
/// See module-level documentation for details.
#[derive(Debug)]
pub struct FIFOCache<K, V>
where
    K: Eq + Hash + Clone,
{
    inner: FIFOCacheInner<K, V>,
    metrics: CacheMetrics,
}

#[derive(Debug)]
pub struct FIFOCacheInner<K, V>
where
    K: Eq + Hash + Clone,
{
    store: HashMapStore<K, V>,
    insertion_order: VecDeque<K>, // Tracks the order of insertion
}

impl<K, V> FIFOCacheInner<K, V>
where
    K: Eq + Hash + Clone,
{
    fn new(capacity: usize) -> Self {
        Self {
            store: HashMapStore::new(capacity),
            insertion_order: VecDeque::with_capacity(capacity),
        }
    }
}

impl<K, V> FIFOCache<K, V>
where
    K: Eq + Hash + Clone,
    V: Debug,
{
    /// Creates a new FIFO cache with the given capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: FIFOCacheInner::new(capacity),
            metrics: CacheMetrics::new(),
        }
    }

    /// Returns the number of items currently in the cache.
    /// This is a duplicate of the CoreCache::len() method but provides direct access.
    pub fn current_size(&self) -> usize {
        self.inner.store.len()
    }

    /// Returns the insertion order length (may include stale entries).
    /// This is primarily for testing and debugging purposes.
    pub fn insertion_order_len(&self) -> usize {
        self.inner.insertion_order.len()
    }

    /// Checks if the internal store contains a specific key.
    /// This is primarily for testing stale entry behavior.
    pub fn cache_contains_key(&self, key: &K) -> bool {
        self.inner.store.contains(key)
    }

    /// Returns an iterator over the insertion order keys.
    /// This is primarily for testing and debugging purposes.
    pub fn insertion_order_iter(&self) -> impl Iterator<Item = &K> {
        self.inner.insertion_order.iter()
    }

    /// Returns snapshot metrics from the cache
    /// Can be used to help understand the state of the cache
    pub fn metrics_snapshot(&self) -> CacheMetricsSnapshot {
        CacheMetricsSnapshot {
            get_calls: self.metrics.get_calls,
            get_hits: self.metrics.get_hits,
            get_misses: self.metrics.get_misses,
            insert_calls: self.metrics.insert_calls,
            insert_updates: self.metrics.insert_updates,
            insert_new: self.metrics.insert_new,
            evict_calls: self.metrics.evict_calls,
            evicted_entries: self.metrics.evicted_entries,
            stale_skips: self.metrics.stale_skips,
            evict_scan_steps: self.metrics.evict_scan_steps,
            pop_oldest_calls: self.metrics.pop_oldest_calls,
            pop_oldest_found: self.metrics.pop_oldest_found,
            pop_oldest_empty_or_stale: self.metrics.pop_oldest_empty_or_stale,
            peek_oldest_calls: self.metrics.peek_oldest_calls.get(),
            peek_oldest_found: self.metrics.peek_oldest_found.get(),
            age_rank_calls: self.metrics.age_rank_calls.get(),
            age_rank_found: self.metrics.age_rank_found.get(),
            age_rank_scan_steps: self.metrics.age_rank_scan_steps.get(),
            cache_len: self.inner.store.len(),
            insertion_order_len: self.inner.insertion_order.len(),
            capacity: self.inner.store.capacity(),
        }
    }

    /// Manually removes a key from the store only (for testing stale entries).
    /// This is only for testing purposes to simulate stale entry conditions.
    /// Accepts a raw key and removes the corresponding entry from the store.
    #[cfg(test)]
    pub fn remove_from_cache_only(&mut self, key: &K) -> Option<Arc<V>> {
        self.inner.store.remove(key)
    }

    /// Returns the current store HashMap capacity (for testing memory usage).
    #[cfg(test)]
    pub fn cache_capacity(&self) -> usize {
        self.inner.store.map_capacity()
    }

    /// Returns the current insertion order VecDeque capacity (for testing memory usage).
    #[cfg(test)]
    pub fn insertion_order_capacity(&self) -> usize {
        self.inner.insertion_order.capacity()
    }

    /// Evicts the oldest valid entry from the cache.
    /// Skips over any stale entries (keys that were lazily deleted).
    fn evict_oldest(&mut self) {
        self.metrics.record_evict_call();
        // Keep popping from the front until we find a valid key or the queue is empty
        while let Some(oldest_key) = self.inner.insertion_order.pop_front() {
            self.metrics.record_evict_scan_step();

            if self.inner.store.contains(&oldest_key) {
                // Found a valid key, remove it and stop
                self.inner.store.remove(&oldest_key);
                self.metrics.record_evicted_entry();
                self.inner.store.record_eviction();
                break;
            }
            // Skip stale entries (keys that were already removed from the cache)
            self.metrics.record_stale_skip();
        }
    }
}

/// Thread-safe FIFO cache wrapper using RwLock.
#[derive(Clone, Debug)]
pub struct ConcurrentFIFOCache<K, V>
where
    K: Eq + Hash + Clone,
{
    inner: Arc<RwLock<FIFOCache<K, V>>>,
}

impl<K, V> ConcurrentFIFOCache<K, V>
where
    K: Eq + Hash + Clone + Debug,
    V: Debug,
{
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(FIFOCache::new(capacity))),
        }
    }

    pub fn insert(&self, key: K, value: V) -> Option<V> {
        let mut cache = self.inner.write();
        cache.insert(key, value)
    }

    pub fn get(&self, key: &K) -> Option<V>
    where
        V: Clone,
    {
        let mut cache = self.inner.write();
        cache.get(key).cloned()
    }

    pub fn contains(&self, key: &K) -> bool {
        let cache = self.inner.read();
        cache.contains(key)
    }

    pub fn len(&self) -> usize {
        let cache = self.inner.read();
        cache.len()
    }

    pub fn is_empty(&self) -> bool {
        let cache = self.inner.read();
        cache.is_empty()
    }

    pub fn capacity(&self) -> usize {
        let cache = self.inner.read();
        cache.capacity()
    }

    pub fn clear(&self) {
        let mut cache = self.inner.write();
        cache.clear();
    }

    pub fn pop_oldest(&self) -> Option<(K, V)> {
        let mut cache = self.inner.write();
        cache.pop_oldest()
    }

    pub fn peek_oldest(&self) -> Option<(K, V)>
    where
        K: Clone,
        V: Clone,
    {
        let cache = self.inner.read();
        cache.peek_oldest().map(|(k, v)| (k.clone(), v.clone()))
    }

    pub fn pop_oldest_batch(&self, count: usize) -> Vec<(K, V)> {
        let mut cache = self.inner.write();
        cache.pop_oldest_batch(count)
    }

    pub fn age_rank(&self, key: &K) -> Option<usize> {
        let cache = self.inner.read();
        cache.age_rank(key)
    }

    pub fn metrics_snapshot(&self) -> CacheMetricsSnapshot {
        let cache = self.inner.read();
        cache.metrics_snapshot()
    }
}

impl<K, V> ConcurrentCache for ConcurrentFIFOCache<K, V>
where
    K: Eq + Hash + Clone + Debug + Send + Sync,
    V: Debug + Send + Sync,
{
}

impl<K, V> MetricsSnapshotProvider<CacheMetricsSnapshot> for FIFOCache<K, V>
where
    K: Eq + Hash + Clone,
    V: Debug,
{
    fn snapshot(&self) -> CacheMetricsSnapshot {
        self.metrics_snapshot()
    }
}

impl<K, V> MetricsSnapshotProvider<CacheMetricsSnapshot> for ConcurrentFIFOCache<K, V>
where
    K: Eq + Hash + Clone + Debug,
    V: Debug,
{
    fn snapshot(&self) -> CacheMetricsSnapshot {
        self.metrics_snapshot()
    }
}

impl<K, V> CoreCache<K, V> for FIFOCache<K, V>
where
    K: Eq + Hash + Clone,
    V: Debug,
{
    fn insert(&mut self, key: K, value: V) -> Option<V> {
        // Update cache metrics
        self.metrics.record_insert_call();

        // If capacity is 0, cannot store anything
        if self.inner.store.capacity() == 0 {
            return None;
        }

        if self.inner.store.contains(&key) {
            self.metrics.record_insert_update();
            if let Ok(previous) = self.inner.store.try_insert(key, Arc::new(value)) {
                return previous.map(|old_value_arc| {
                    Arc::try_unwrap(old_value_arc).expect("external Arc<V> references detected")
                });
            }
            return None;
        }

        self.metrics.record_insert_new();

        // If the cache is at capacity, remove the oldest valid item (FIFO)
        if self.inner.store.len() >= self.inner.store.capacity() {
            self.evict_oldest();
        }

        // Add the new key to the insertion order and cache
        let key_for_queue = key.clone();
        if self.inner.store.try_insert(key, Arc::new(value)).is_ok() {
            self.inner.insertion_order.push_back(key_for_queue);
        }
        None
    }

    fn get(&mut self, key: &K) -> Option<&V> {
        // In FIFO, getting an item doesn't change its position
        match self.inner.store.get_ref(key) {
            Some(value) => {
                self.metrics.record_get_hit();
                Some(value.as_ref())
            },
            None => {
                self.metrics.record_get_miss();
                None
            },
        }
    }

    fn contains(&self, key: &K) -> bool {
        self.inner.store.contains(key)
    }

    fn len(&self) -> usize {
        self.inner.store.len()
    }

    fn capacity(&self) -> usize {
        self.inner.store.capacity()
    }

    fn clear(&mut self) {
        self.inner.store.clear();
        self.inner.insertion_order.clear();
    }
}

impl<K, V> FIFOCacheTrait<K, V> for FIFOCache<K, V>
where
    K: Eq + Hash + Clone + Debug,
    V: Debug,
{
    fn pop_oldest(&mut self) -> Option<(K, V)> {
        self.metrics.record_pop_oldest_call();

        // Use the existing evict_oldest logic but return the key-value pair
        while let Some(oldest_key) = self.inner.insertion_order.pop_front() {
            if let Some(value_arc) = self.inner.store.remove(&oldest_key) {
                self.metrics.record_pop_oldest_found();
                self.inner.store.record_eviction();

                // Try to unwrap both Arcs to get the original key and value
                let value =
                    Arc::try_unwrap(value_arc).expect("external Arc<V> references detected");

                return Some((oldest_key, value));
            }
            // Skip stale entries (keys that were already removed from the cache)
        }
        self.metrics.record_pop_oldest_empty_or_stale();
        None
    }

    fn peek_oldest(&self) -> Option<(&K, &V)> {
        (&self.metrics).record_peek_oldest_call();

        // Find the first valid entry in the insertion order
        for key in &self.inner.insertion_order {
            if let Some(value_arc) = self.inner.store.peek_ref(key) {
                (&self.metrics).record_peek_oldest_found();
                return Some((key, value_arc.as_ref()));
            }
        }
        None
    }

    fn pop_oldest_batch(&mut self, count: usize) -> Vec<(K, V)> {
        let mut result = Vec::with_capacity(count.min(<Self as CoreCache<K, V>>::len(self)));
        for _ in 0..count {
            if let Some(entry) = self.pop_oldest() {
                result.push(entry);
            } else {
                break;
            }
        }
        result
    }

    fn age_rank(&self, key: &K) -> Option<usize> {
        (&self.metrics).record_age_rank_call();
        // Find position in insertion order, accounting for stale entries
        let mut rank = 0;
        for insertion_key in &self.inner.insertion_order {
            (&self.metrics).record_age_rank_scan_step();
            if self.inner.store.contains(insertion_key) {
                if insertion_key == key {
                    (&self.metrics).record_age_rank_found();
                    return Some(rank);
                }
                rank += 1;
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use crate::policy::fifo::FIFOCache;
    use crate::traits::{CoreCache, FIFOCacheTrait};
    use std::collections::HashSet;

    // Basic FIFO Behavior Tests
    mod basic_behavior {
        use super::*;
        use crate::traits::FIFOCacheTrait;

        #[test]
        fn test_basic_fifo_insertion_and_retrieval() {
            let mut cache = FIFOCache::new(3);

            // Test basic insertion and retrieval
            assert_eq!(cache.insert("key1", "value1"), None);
            assert_eq!(cache.insert("key2", "value2"), None);
            assert_eq!(cache.insert("key3", "value3"), None);

            assert_eq!(cache.get(&"key1"), Some(&"value1"));
            assert_eq!(cache.get(&"key2"), Some(&"value2"));
            assert_eq!(cache.get(&"key3"), Some(&"value3"));
            assert_eq!(cache.len(), 3);
        }

        #[test]
        fn test_fifo_eviction_order() {
            let mut cache = FIFOCache::new(2);

            // Fill cache to capacity
            cache.insert("first", "value1");
            cache.insert("second", "value2");
            assert_eq!(cache.len(), 2);

            // Insert a third item - should evict "first" (oldest)
            cache.insert("third", "value3");
            assert_eq!(cache.len(), 2);
            assert!(!cache.contains(&"first")); // The first item should be evicted
            assert!(cache.contains(&"second"));
            assert!(cache.contains(&"third"));
        }

        #[test]
        fn test_capacity_enforcement() {
            let mut cache = FIFOCache::new(3);

            // Fill beyond capacity
            for i in 1..=5 {
                cache.insert(format!("key{}", i), format!("value{}", i));
            }

            // Should only contain the last 3 items due to FIFO eviction
            assert_eq!(cache.len(), 3);
            assert!(!cache.contains(&"key1".to_string()));
            assert!(!cache.contains(&"key2".to_string()));
            assert!(cache.contains(&"key3".to_string()));
            assert!(cache.contains(&"key4".to_string()));
            assert!(cache.contains(&"key5".to_string()));
        }

        #[test]
        fn test_update_existing_key() {
            let mut cache = FIFOCache::new(3);

            cache.insert("key1", "original");
            cache.insert("key2", "value2");

            // Update the existing key - should return an old value
            let old_value = cache.insert("key1", "updated");
            assert_eq!(old_value, Some("original"));
            assert_eq!(cache.get(&"key1"), Some(&"updated"));
            assert_eq!(cache.len(), 2); // Length shouldn't change
        }

        #[test]
        fn test_insertion_order_preservation() {
            let mut cache = FIFOCache::new(4);

            // Insert items in a specific order
            cache.insert("first", 1);
            cache.insert("second", 2);
            cache.insert("third", 3);
            cache.insert("fourth", 4);

            // Verify insertion order is preserved in FIFO operations
            assert_eq!(cache.peek_oldest(), Some((&"first", &1)));
            assert_eq!(cache.age_rank(&"first"), Some(0)); // oldest
            assert_eq!(cache.age_rank(&"second"), Some(1));
            assert_eq!(cache.age_rank(&"third"), Some(2));
            assert_eq!(cache.age_rank(&"fourth"), Some(3)); // newest

            // Pop oldest and verify order shifts correctly
            assert_eq!(cache.pop_oldest(), Some(("first", 1)));
            assert_eq!(cache.peek_oldest(), Some((&"second", &2)));
            assert_eq!(cache.age_rank(&"second"), Some(0)); // now oldest
            assert_eq!(cache.age_rank(&"third"), Some(1));
            assert_eq!(cache.age_rank(&"fourth"), Some(2)); // still newest

            // Insert a new item and verify it becomes newest
            cache.insert("fifth", 5);
            assert_eq!(cache.age_rank(&"second"), Some(0));
            assert_eq!(cache.age_rank(&"third"), Some(1));
            assert_eq!(cache.age_rank(&"fourth"), Some(2));
            assert_eq!(cache.age_rank(&"fifth"), Some(3)); // newest

            // Test eviction maintains order - add item beyond capacity
            cache.insert("sixth", 6);
            // Should evict "second" (oldest)
            assert!(!cache.contains(&"second"));
            assert_eq!(cache.peek_oldest(), Some((&"third", &3)));
            assert_eq!(cache.age_rank(&"third"), Some(0)); // now oldest
            assert_eq!(cache.age_rank(&"fourth"), Some(1));
            assert_eq!(cache.age_rank(&"fifth"), Some(2));
            assert_eq!(cache.age_rank(&"sixth"), Some(3)); // newest
        }

        #[test]
        fn test_key_operations_consistency() {
            let mut cache = FIFOCache::new(3);

            // Test consistency between contents, get, and len
            assert_eq!(cache.len(), 0);
            assert!(!cache.contains(&"key1"));
            assert_eq!(cache.get(&"key1"), None);

            // Insert first item
            cache.insert("key1", "value1");
            assert_eq!(cache.len(), 1);
            assert!(cache.contains(&"key1"));
            assert_eq!(cache.get(&"key1"), Some(&"value1"));
            assert!(!cache.contains(&"key2"));
            assert_eq!(cache.get(&"key2"), None);

            // Insert the second item
            cache.insert("key2", "value2");
            assert_eq!(cache.len(), 2);
            assert!(cache.contains(&"key1"));
            assert!(cache.contains(&"key2"));
            assert_eq!(cache.get(&"key1"), Some(&"value1"));
            assert_eq!(cache.get(&"key2"), Some(&"value2"));

            // Update existing key - len should not change
            let old_value = cache.insert("key1", "updated_value1");
            assert_eq!(old_value, Some("value1"));
            assert_eq!(cache.len(), 2); // Should remain 2
            assert!(cache.contains(&"key1"));
            assert_eq!(cache.get(&"key1"), Some(&"updated_value1"));
            assert_eq!(cache.get(&"key2"), Some(&"value2"));

            // Fill to capacity
            cache.insert("key3", "value3");
            assert_eq!(cache.len(), 3);
            assert_eq!(cache.capacity(), 3);
            assert!(cache.contains(&"key1"));
            assert!(cache.contains(&"key2"));
            assert!(cache.contains(&"key3"));

            // Trigger eviction - should evict key1 (oldest, even though updated)
            // In true FIFO, updating a key does NOT change its insertion order position
            cache.insert("key4", "value4");
            assert_eq!(cache.len(), 3); // Should remain at capacity
            assert!(!cache.contains(&"key1")); // Should be evicted (oldest insertion time)
            assert!(cache.contains(&"key2")); // Should remain
            assert!(cache.contains(&"key3"));
            assert!(cache.contains(&"key4"));
            assert_eq!(cache.get(&"key1"), None);
            assert_eq!(cache.get(&"key4"), Some(&"value4"));

            // Test pop_oldest consistency - should pop key2 (now oldest)
            let popped = cache.pop_oldest();
            assert_eq!(popped, Some(("key2", "value2")));
            assert_eq!(cache.len(), 2);
            assert!(!cache.contains(&"key2"));
            assert_eq!(cache.get(&"key2"), None);

            // Verify remaining items
            assert!(cache.contains(&"key3"));
            assert!(cache.contains(&"key4"));

            // Clear and verify all operations report empty state
            cache.clear();
            assert_eq!(cache.len(), 0);
            assert!(!cache.contains(&"key2"));
            assert!(!cache.contains(&"key3"));
            assert!(!cache.contains(&"key4"));
            assert_eq!(cache.get(&"key2"), None);
            assert_eq!(cache.get(&"key3"), None);
            assert_eq!(cache.get(&"key4"), None);
            assert_eq!(cache.peek_oldest(), None);
            assert_eq!(cache.pop_oldest(), None);
        }

        #[test]
        fn test_basic_fifo_metrics() {
            let mut cache = FIFOCache::new(3);

            // Test basic insertion and retrieval
            assert_eq!(cache.insert("key1", "value1"), None);
            assert_eq!(cache.insert("key2", "value2"), None);
            assert_eq!(cache.insert("key3", "value3"), None);

            assert_eq!(cache.get(&"key1"), Some(&"value1"));
            assert_eq!(cache.get(&"key2"), Some(&"value2"));
            assert_eq!(cache.get(&"key3"), Some(&"value3"));
            assert_eq!(cache.len(), 3);

            assert_eq!(cache.metrics.insert_new, 3);
            assert_eq!(cache.metrics.get_calls, 3);
        }
    }

    // Edge Cases Tests
    mod edge_cases {
        use super::*;

        #[test]
        fn test_empty_cache_operations() {
            let mut cache: FIFOCache<String, String> = FIFOCache::new(5);

            assert_eq!(cache.get(&"nonexistent".to_string()), None);
            assert!(!cache.contains(&"nonexistent".to_string()));
            assert_eq!(cache.len(), 0);
            assert_eq!(cache.capacity(), 5);

            // Clear empty cache should work
            cache.clear();
            assert_eq!(cache.len(), 0);
        }

        #[test]
        fn test_single_item_cache() {
            let mut cache = FIFOCache::new(1);

            cache.insert("only", "value1");
            assert_eq!(cache.len(), 1);
            assert_eq!(cache.get(&"only"), Some(&"value1"));

            // Insert the second item - should evict the first
            cache.insert("second", "value2");
            assert_eq!(cache.len(), 1);
            assert!(!cache.contains(&"only"));
            assert!(cache.contains(&"second"));
        }

        #[test]
        fn test_zero_capacity_cache() {
            let mut cache = FIFOCache::new(0);

            // Should not be able to store anything
            cache.insert("key", "value");
            assert_eq!(cache.len(), 0);
            assert!(!cache.contains(&"key"));
            assert_eq!(cache.get(&"key"), None);
        }

        #[test]
        fn test_clear_operation() {
            let mut cache = FIFOCache::new(3);

            cache.insert("key1", "value1");
            cache.insert("key2", "value2");
            assert_eq!(cache.len(), 2);

            cache.clear();
            assert_eq!(cache.len(), 0);
            assert!(!cache.contains(&"key1"));
            assert!(!cache.contains(&"key2"));
            assert_eq!(cache.get(&"key1"), None);
        }

        #[test]
        fn test_duplicate_key_handling() {
            let mut cache = FIFOCache::new(3);

            // Insert the initial key
            assert_eq!(cache.insert("key1", "value1"), None);
            assert_eq!(cache.len(), 1);
            assert!(cache.contains(&"key1"));
            assert_eq!(cache.get(&"key1"), Some(&"value1"));

            // Insert the same key again - should update, not add a new entry
            assert_eq!(cache.insert("key1", "value1_updated"), Some("value1"));
            assert_eq!(cache.len(), 1); // Length should remain the same
            assert_eq!(cache.get(&"key1"), Some(&"value1_updated"));

            // Add more keys
            cache.insert("key2", "value2");
            cache.insert("key3", "value3");
            assert_eq!(cache.len(), 3);

            // Update the existing key multiple times
            assert_eq!(cache.insert("key2", "value2_v2"), Some("value2"));
            assert_eq!(cache.insert("key2", "value2_v3"), Some("value2_v2"));
            assert_eq!(cache.len(), 3); // Should still be 3
            assert_eq!(cache.get(&"key2"), Some(&"value2_v3"));

            // Verify insertion order preserved (key1 should still be oldest)
            assert_eq!(cache.age_rank(&"key1"), Some(0)); // Still oldest despite updates
            assert_eq!(cache.age_rank(&"key2"), Some(1));
            assert_eq!(cache.age_rank(&"key3"), Some(2));

            // Fill to capacity and force eviction
            cache.insert("key4", "value4");
            assert_eq!(cache.len(), 3);
            assert!(!cache.contains(&"key1")); // Oldest should be evicted
            assert!(cache.contains(&"key2"));
            assert!(cache.contains(&"key3"));
            assert!(cache.contains(&"key4"));

            // Update existing key after eviction
            assert_eq!(cache.insert("key2", "value2_final"), Some("value2_v3"));
            assert_eq!(cache.len(), 3);
            assert_eq!(cache.get(&"key2"), Some(&"value2_final"));

            // Verify no duplicate entries in internal structures
            let mut found_keys = HashSet::new();
            let mut count = 0;
            while let Some((key, _)) = cache.pop_oldest() {
                assert!(found_keys.insert(key), "Duplicate key found: {}", key);
                count += 1;
            }
            assert_eq!(count, 3); // Should have exactly 3 unique keys
        }

        #[test]
        fn test_boundary_conditions() {
            let mut cache = FIFOCache::new(2);

            // Test exactly at capacity
            cache.insert("key1", "value1");
            assert_eq!(cache.len(), 1);
            assert!(!cache.contains(&"key2"));

            cache.insert("key2", "value2");
            assert_eq!(cache.len(), 2); // Exactly at capacity
            assert_eq!(cache.capacity(), 2);
            assert!(cache.contains(&"key1"));
            assert!(cache.contains(&"key2"));

            // Test operations at capacity limit
            assert_eq!(cache.peek_oldest(), Some((&"key1", &"value1")));
            assert_eq!(cache.age_rank(&"key1"), Some(0));
            assert_eq!(cache.age_rank(&"key2"), Some(1));

            // Insert one more to trigger eviction (capacity + 1)
            cache.insert("key3", "value3");
            assert_eq!(cache.len(), 2); // Should remain at capacity
            assert!(!cache.contains(&"key1")); // Oldest evicted
            assert!(cache.contains(&"key2"));
            assert!(cache.contains(&"key3"));

            // Test boundary with updates at capacity
            assert_eq!(cache.insert("key2", "value2_updated"), Some("value2"));
            assert_eq!(cache.len(), 2); // Still at capacity
            assert_eq!(cache.get(&"key2"), Some(&"value2_updated"));

            // Test pop operations at the boundary
            assert_eq!(cache.pop_oldest(), Some(("key2", "value2_updated")));
            assert_eq!(cache.len(), 1); // One below capacity

            assert_eq!(cache.pop_oldest(), Some(("key3", "value3")));
            assert_eq!(cache.len(), 0); // Empty

            // Test operations on empty cache after reaching boundary
            assert_eq!(cache.pop_oldest(), None);
            assert_eq!(cache.peek_oldest(), None);
            assert_eq!(cache.len(), 0);

            // Test filling back to capacity
            cache.insert("new1", "new_val1");
            cache.insert("new2", "new_val2");
            assert_eq!(cache.len(), 2); // Back to capacity

            // Test batch operations at the boundary
            let batch = cache.pop_oldest_batch(3); // Request more than available
            assert_eq!(batch.len(), 2); // Should get exactly what's available
            assert_eq!(cache.len(), 0); // Should be empty
        }

        #[test]
        fn test_empty_to_full_transition() {
            let mut cache = FIFOCache::new(4);

            // Start empty
            assert_eq!(cache.len(), 0);
            assert_eq!(cache.capacity(), 4);
            assert_eq!(cache.peek_oldest(), None);
            assert_eq!(cache.pop_oldest(), None);

            // Fill step by step, verifying the state at each step

            // Step 1: Insert first item
            cache.insert("item1", 1);
            assert_eq!(cache.len(), 1);
            assert!(cache.contains(&"item1"));
            assert_eq!(cache.peek_oldest(), Some((&"item1", &1)));
            assert_eq!(cache.age_rank(&"item1"), Some(0));

            // Step 2: Insert a second item
            cache.insert("item2", 2);
            assert_eq!(cache.len(), 2);
            assert!(cache.contains(&"item1"));
            assert!(cache.contains(&"item2"));
            assert_eq!(cache.peek_oldest(), Some((&"item1", &1))); // Still oldest
            assert_eq!(cache.age_rank(&"item1"), Some(0));
            assert_eq!(cache.age_rank(&"item2"), Some(1));

            // Step 3: Insert a third item
            cache.insert("item3", 3);
            assert_eq!(cache.len(), 3);
            assert_eq!(cache.peek_oldest(), Some((&"item1", &1)));
            assert_eq!(cache.age_rank(&"item1"), Some(0));
            assert_eq!(cache.age_rank(&"item2"), Some(1));
            assert_eq!(cache.age_rank(&"item3"), Some(2));

            // Step 4: Fill to capacity
            cache.insert("item4", 4);
            assert_eq!(cache.len(), 4); // Now at full capacity
            assert_eq!(cache.capacity(), 4);

            // Verify all items present and in correct order
            assert!(cache.contains(&"item1"));
            assert_eq!(cache.get(&"item1"), Some(&1));
            assert_eq!(cache.age_rank(&"item1"), Some(0));

            assert!(cache.contains(&"item2"));
            assert_eq!(cache.get(&"item2"), Some(&2));
            assert_eq!(cache.age_rank(&"item2"), Some(1));

            assert!(cache.contains(&"item3"));
            assert_eq!(cache.get(&"item3"), Some(&3));
            assert_eq!(cache.age_rank(&"item3"), Some(2));

            assert!(cache.contains(&"item4"));
            assert_eq!(cache.get(&"item4"), Some(&4));
            assert_eq!(cache.age_rank(&"item4"), Some(3));

            // Verify oldest is still first
            assert_eq!(cache.peek_oldest(), Some((&"item1", &1)));

            // Test that we're truly at capacity - the next insert should evict
            cache.insert("item5", 5);
            assert_eq!(cache.len(), 4); // Still at capacity
            assert!(!cache.contains(&"item1")); // First item evicted
            assert!(cache.contains(&"item5")); // New item added
            assert_eq!(cache.peek_oldest(), Some((&"item2", &2))); // item2 now oldest
        }

        #[test]
        fn test_full_to_empty_transition() {
            // Helper function to create cache with same initial state (avoids cloning)
            let create_test_cache = || {
                let mut cache = FIFOCache::new(3);
                cache.insert("item1", 1);
                cache.insert("item2", 2);
                cache.insert("item3", 3);
                assert_eq!(cache.len(), 3);
                assert_eq!(cache.capacity(), 3);
                cache
            };

            // Method 1: Empty using pop_oldest one by one
            let mut emptying_cache = create_test_cache();

            // First pop
            assert_eq!(emptying_cache.pop_oldest(), Some(("item1", 1)));
            assert_eq!(emptying_cache.len(), 2);
            assert!(!emptying_cache.contains(&"item1"));
            assert!(emptying_cache.contains(&"item2"));
            assert!(emptying_cache.contains(&"item3"));
            assert_eq!(emptying_cache.peek_oldest(), Some((&"item2", &2)));

            // Second pop
            assert_eq!(emptying_cache.pop_oldest(), Some(("item2", 2)));
            assert_eq!(emptying_cache.len(), 1);
            assert!(!emptying_cache.contains(&"item2"));
            assert!(emptying_cache.contains(&"item3"));
            assert_eq!(emptying_cache.peek_oldest(), Some((&"item3", &3)));

            // Third pop
            assert_eq!(emptying_cache.pop_oldest(), Some(("item3", 3)));
            assert_eq!(emptying_cache.len(), 0);
            assert!(!emptying_cache.contains(&"item3"));
            assert_eq!(emptying_cache.peek_oldest(), None);

            // Fourth pop on empty cache
            assert_eq!(emptying_cache.pop_oldest(), None);
            assert_eq!(emptying_cache.len(), 0);

            // Method 2: Empty using batch operation
            let mut batch_cache = create_test_cache();
            let batch = batch_cache.pop_oldest_batch(3);
            assert_eq!(batch.len(), 3);
            assert_eq!(batch[0], ("item1", 1));
            assert_eq!(batch[1], ("item2", 2));
            assert_eq!(batch[2], ("item3", 3));
            assert_eq!(batch_cache.len(), 0);
            assert_eq!(batch_cache.peek_oldest(), None);

            // Method 3: Empty using a clear operation
            let mut clear_cache = create_test_cache();
            clear_cache.clear();
            assert_eq!(clear_cache.len(), 0);
            assert_eq!(clear_cache.capacity(), 3); // Capacity unchanged
            assert_eq!(clear_cache.peek_oldest(), None);
            assert_eq!(clear_cache.pop_oldest(), None);

            // Verify cache can be refilled after each emptying method
            for mut test_cache in [emptying_cache, batch_cache, clear_cache] {
                test_cache.insert("new1", 100);
                assert_eq!(test_cache.len(), 1);
                assert!(test_cache.contains(&"new1"));
                assert_eq!(test_cache.peek_oldest(), Some((&"new1", &100)));
            }

            // Test partial emptying and refilling
            let mut partial_cache = FIFOCache::new(4);
            partial_cache.insert("a", 1);
            partial_cache.insert("b", 2);
            partial_cache.insert("c", 3);
            partial_cache.insert("d", 4);

            // Remove 2 items
            partial_cache.pop_oldest(); // Remove "a"
            partial_cache.pop_oldest(); // Remove "b"
            assert_eq!(partial_cache.len(), 2);

            // Add 2 new items
            partial_cache.insert("e", 5);
            partial_cache.insert("f", 6);
            assert_eq!(partial_cache.len(), 4); // Back to full

            // Verify the correct order maintained
            assert_eq!(partial_cache.peek_oldest(), Some((&"c", &3))); // "c" should be oldest
            assert_eq!(partial_cache.age_rank(&"c"), Some(0));
            assert_eq!(partial_cache.age_rank(&"d"), Some(1));
            assert_eq!(partial_cache.age_rank(&"e"), Some(2));
            assert_eq!(partial_cache.age_rank(&"f"), Some(3));
        }
    }

    // FIFO Trait Methods Tests
    mod trait_methods {
        use super::*;

        #[test]
        fn test_pop_oldest() {
            let mut cache = FIFOCache::new(3);

            cache.insert("first", "value1");
            cache.insert("second", "value2");
            cache.insert("third", "value3");

            // Pop oldest should return first inserted
            assert_eq!(cache.pop_oldest(), Some(("first", "value1")));
            assert_eq!(cache.len(), 2);
            assert!(!cache.contains(&"first"));

            // The next pop should return second oldest
            assert_eq!(cache.pop_oldest(), Some(("second", "value2")));
            assert_eq!(cache.len(), 1);
        }

        #[test]
        fn test_peek_oldest() {
            let mut cache = FIFOCache::new(3);

            cache.insert("first", "value1");
            cache.insert("second", "value2");

            // Peek should return oldest without removing
            assert_eq!(cache.peek_oldest(), Some((&"first", &"value1")));
            assert_eq!(cache.len(), 2); // Should not change length
            assert!(cache.contains(&"first")); // Should still be present

            // Multiple peeks should return the same result
            assert_eq!(cache.peek_oldest(), Some((&"first", &"value1")));
        }

        #[test]
        fn test_age_rank() {
            let mut cache = FIFOCache::new(4);

            cache.insert("first", "value1"); // rank 0 (oldest)
            cache.insert("second", "value2"); // rank 1
            cache.insert("third", "value3"); // rank 2
            cache.insert("fourth", "value4"); // rank 3 (newest)

            assert_eq!(cache.age_rank(&"first"), Some(0));
            assert_eq!(cache.age_rank(&"second"), Some(1));
            assert_eq!(cache.age_rank(&"third"), Some(2));
            assert_eq!(cache.age_rank(&"fourth"), Some(3));
            assert_eq!(cache.age_rank(&"nonexistent"), None);
        }

        #[test]
        fn test_pop_oldest_batch() {
            let mut cache = FIFOCache::new(5);

            for i in 1..=5 {
                cache.insert(format!("key{}", i), format!("value{}", i));
            }

            // Pop a batch of 3 items
            let batch = cache.pop_oldest_batch(3);
            assert_eq!(batch.len(), 3);
            assert_eq!(batch[0], ("key1".to_string(), "value1".to_string()));
            assert_eq!(batch[1], ("key2".to_string(), "value2".to_string()));
            assert_eq!(batch[2], ("key3".to_string(), "value3".to_string()));

            // Cache should have 2 items left
            assert_eq!(cache.len(), 2);
            assert!(cache.contains(&"key4".to_string()));
            assert!(cache.contains(&"key5".to_string()));
        }

        #[test]
        fn test_pop_oldest_batch_more_than_available() {
            let mut cache = FIFOCache::new(3);

            cache.insert("key1", "value1");
            cache.insert("key2", "value2");

            // Request more than available
            let batch = cache.pop_oldest_batch(5);
            assert_eq!(batch.len(), 2); // Should only return what's available
            assert_eq!(cache.len(), 0); // Cache should be empty
        }

        #[test]
        fn test_pop_oldest_empty_cache() {
            let mut cache: FIFOCache<String, String> = FIFOCache::new(5);

            // Pop from the empty cache should return None
            assert_eq!(cache.pop_oldest(), None);
            assert_eq!(cache.len(), 0);

            // Multiple pops should still return None
            assert_eq!(cache.pop_oldest(), None);
            assert_eq!(cache.pop_oldest(), None);
            assert_eq!(cache.len(), 0);

            // Add one item, pop it, then pop from empty again
            cache.insert("key1".to_string(), "value1".to_string());
            assert_eq!(cache.len(), 1);

            let popped = cache.pop_oldest();
            assert_eq!(popped, Some(("key1".to_string(), "value1".to_string())));
            assert_eq!(cache.len(), 0);

            // Now it's empty again
            assert_eq!(cache.pop_oldest(), None);
            assert_eq!(cache.len(), 0);
        }

        #[test]
        fn test_peek_oldest_empty_cache() {
            let cache: FIFOCache<String, String> = FIFOCache::new(5);

            // Peek at the empty cache should return None
            assert_eq!(cache.peek_oldest(), None);
            assert_eq!(cache.len(), 0);

            // Multiple peeks should still return None
            assert_eq!(cache.peek_oldest(), None);
            assert_eq!(cache.peek_oldest(), None);
            assert_eq!(cache.len(), 0);

            // Test peek after clear
            let mut test_cache = FIFOCache::new(3);
            test_cache.insert("key1".to_string(), "value1".to_string());
            test_cache.insert("key2".to_string(), "value2".to_string());

            // Should have content
            assert!(test_cache.peek_oldest().is_some());
            assert_eq!(test_cache.len(), 2);

            // Clear and peek again
            test_cache.clear();
            assert_eq!(test_cache.peek_oldest(), None);
            assert_eq!(test_cache.len(), 0);
        }

        #[test]
        fn test_age_rank_after_eviction() {
            let mut cache = FIFOCache::new(3);

            // Fill cache
            cache.insert("first", 1);
            cache.insert("second", 2);
            cache.insert("third", 3);

            // Verify initial ranks
            assert_eq!(cache.age_rank(&"first"), Some(0));
            assert_eq!(cache.age_rank(&"second"), Some(1));
            assert_eq!(cache.age_rank(&"third"), Some(2));

            // Trigger eviction by adding a fourth item
            cache.insert("fourth", 4);

            // "first" should be evicted, ranks should shift
            assert_eq!(cache.age_rank(&"first"), None); // Evicted
            assert_eq!(cache.age_rank(&"second"), Some(0)); // Now oldest
            assert_eq!(cache.age_rank(&"third"), Some(1));
            assert_eq!(cache.age_rank(&"fourth"), Some(2)); // Newest

            // Add another item to trigger another eviction
            cache.insert("fifth", 5);

            // "second" should be evicted, ranks shift again
            assert_eq!(cache.age_rank(&"first"), None); // Still evicted
            assert_eq!(cache.age_rank(&"second"), None); // Now evicted
            assert_eq!(cache.age_rank(&"third"), Some(0)); // Now oldest
            assert_eq!(cache.age_rank(&"fourth"), Some(1));
            assert_eq!(cache.age_rank(&"fifth"), Some(2)); // Newest

            // Test manual eviction with pop_oldest
            let popped = cache.pop_oldest();
            assert_eq!(popped, Some(("third", 3)));

            // Ranks should shift after manual pop
            assert_eq!(cache.age_rank(&"third"), None); // Just popped
            assert_eq!(cache.age_rank(&"fourth"), Some(0)); // Now oldest
            assert_eq!(cache.age_rank(&"fifth"), Some(1)); // Now newest

            // Test batch eviction
            cache.insert("sixth", 6);
            cache.insert("seventh", 7);
            assert_eq!(cache.len(), 3);

            let batch = cache.pop_oldest_batch(2);
            assert_eq!(batch.len(), 2);
            assert_eq!(cache.len(), 1);

            // Only "seventh" should remain
            assert_eq!(cache.age_rank(&"fourth"), None);
            assert_eq!(cache.age_rank(&"fifth"), None);
            assert_eq!(cache.age_rank(&"sixth"), None);
            assert_eq!(cache.age_rank(&"seventh"), Some(0)); // Only remaining item
        }

        #[test]
        fn test_batch_operations_edge_cases() {
            let mut cache = FIFOCache::new(3);

            // Test batch with count = 0
            cache.insert("key1", "value1");
            cache.insert("key2", "value2");

            let batch = cache.pop_oldest_batch(0);
            assert_eq!(batch.len(), 0);
            assert_eq!(cache.len(), 2); // Nothing should be removed
            assert!(cache.contains(&"key1"));
            assert!(cache.contains(&"key2"));

            // Test batch on empty cache
            let mut empty_cache: FIFOCache<String, String> = FIFOCache::new(5);
            let empty_batch = empty_cache.pop_oldest_batch(3);
            assert_eq!(empty_batch.len(), 0);
            assert_eq!(empty_cache.len(), 0);

            // Test batch with count = 1 (single item)
            let batch_one = cache.pop_oldest_batch(1);
            assert_eq!(batch_one.len(), 1);
            assert_eq!(batch_one[0], ("key1", "value1"));
            assert_eq!(cache.len(), 1);
            assert!(!cache.contains(&"key1"));
            assert!(cache.contains(&"key2"));

            // Test batch equal to cache size
            cache.insert("key3", "value3");
            cache.insert("key4", "value4");
            assert_eq!(cache.len(), 3);

            let all_batch = cache.pop_oldest_batch(3);
            assert_eq!(all_batch.len(), 3);
            assert_eq!(all_batch[0], ("key2", "value2"));
            assert_eq!(all_batch[1], ("key3", "value3"));
            assert_eq!(all_batch[2], ("key4", "value4"));
            assert_eq!(cache.len(), 0);

            // Verify the cache is completely empty
            assert_eq!(cache.peek_oldest(), None);
            assert_eq!(cache.pop_oldest(), None);

            // Test large batch request on small cache
            cache.insert("only", "item");
            let large_batch = cache.pop_oldest_batch(100);
            assert_eq!(large_batch.len(), 1);
            assert_eq!(large_batch[0], ("only", "item"));
            assert_eq!(cache.len(), 0);

            // Test batch on cache that became empty during operation
            cache.insert("a", "1");
            cache.insert("b", "2");

            // First, empty the cache manually
            cache.clear();

            // Then try batch operation on now-empty cache
            let post_clear_batch = cache.pop_oldest_batch(5);
            assert_eq!(post_clear_batch.len(), 0);
            assert_eq!(cache.len(), 0);
        }
    }

    // Stale Entry Handling Tests
    mod stale_entries {
        use super::*;

        #[test]
        fn test_stale_entry_skipping_during_eviction() {
            let mut cache = FIFOCache::new(3);

            // Fill cache to capacity
            cache.insert("key1", "value1");
            cache.insert("key2", "value2");
            cache.insert("key3", "value3");
            assert_eq!(cache.len(), 3);

            // Manually remove key1 from HashMap but leave it in insertion_order
            // This simulates how stale entries would occur in a more complex scenario
            cache.remove_from_cache_only(&"key1");
            assert_eq!(cache.len(), 2); // HashMap now has 2 items

            // The insertion_order VecDeque still has 3 entries, but "key1" is now stale
            assert_eq!(cache.insertion_order_len(), 3);
            assert!(!cache.contains(&"key1")); // key1 is not in the cache anymore
            assert!(cache.contains(&"key2")); // key2 is still valid
            assert!(cache.contains(&"key3")); // key3 is still valid

            // Add another item to get back to capacity (so the next insert will trigger eviction)
            cache.insert("temp", "temp_value");
            assert_eq!(cache.len(), 3);

            // Now trigger eviction by inserting a new item
            // This should skip over the stale "key1" entry and evict "key2" instead
            cache.insert("key4", "value4");

            // Verifies the eviction skipped stale entry and evicted the next valid entry
            assert_eq!(cache.len(), 3); // Should remain at capacity
            assert!(!cache.contains(&"key1")); // Still not present (was stale)
            assert!(!cache.contains(&"key2")); // Should be evicted (oldest valid)
            assert!(cache.contains(&"key3")); // Should remain
            assert!(cache.contains(&"temp")); // Should remain
            assert!(cache.contains(&"key4")); // Should be newly added

            // Test that further operations continue to work correctly
            cache.insert("key5", "value5");
            assert_eq!(cache.len(), 3);
            assert!(!cache.contains(&"key3")); // key3 should be evicted now
            assert!(cache.contains(&"temp"));
            assert!(cache.contains(&"key4"));
            assert!(cache.contains(&"key5"));
        }

        #[test]
        fn test_insertion_order_consistency_with_stale_entries() {
            let mut cache = FIFOCache::new(4);

            // Fill cache
            cache.insert("a", 1);
            cache.insert("b", 2);
            cache.insert("c", 3);
            cache.insert("d", 4);

            // Verify initial age ranks
            assert_eq!(cache.age_rank(&"a"), Some(0));
            assert_eq!(cache.age_rank(&"b"), Some(1));
            assert_eq!(cache.age_rank(&"c"), Some(2));
            assert_eq!(cache.age_rank(&"d"), Some(3));

            // Manually create stale entries by removing from HashMap
            cache.remove_from_cache_only(&"a");
            cache.remove_from_cache_only(&"c");

            // Now "a" and "c" are stale entries in insertion_order
            assert_eq!(cache.len(), 2); // Only "b" and "d" remain in HashMap
            assert_eq!(cache.insertion_order_len(), 4); // All 4 still in insertion_order

            // age_rank should skip stale entries and give correct ranks for valid entries
            assert_eq!(cache.age_rank(&"a"), None); // Stale, should return None
            assert_eq!(cache.age_rank(&"b"), Some(0)); // First valid entry
            assert_eq!(cache.age_rank(&"c"), None); // Stale, should return None
            assert_eq!(cache.age_rank(&"d"), Some(1)); // Second valid entry

            // peek_oldest should skip stale entries and return the oldest valid entry
            assert_eq!(cache.peek_oldest(), Some((&"b", &2)));

            // pop_oldest should skip stale entries and pop the oldest valid entry
            assert_eq!(cache.pop_oldest(), Some(("b", 2)));

            // After popping "b", stale entries should be cleaned up
            // and "d" should now be the oldest (and only) valid entry
            assert_eq!(cache.len(), 1);
            assert_eq!(cache.age_rank(&"d"), Some(0));
            assert_eq!(cache.peek_oldest(), Some((&"d", &4)));

            // Add new items and verify order is maintained
            cache.insert("e", 5);
            cache.insert("f", 6);

            assert_eq!(cache.age_rank(&"d"), Some(0)); // Still oldest
            assert_eq!(cache.age_rank(&"e"), Some(1));
            assert_eq!(cache.age_rank(&"f"), Some(2));
        }

        #[test]
        fn test_lazy_deletion_behavior() {
            let mut cache = FIFOCache::new(3);

            // Test 1: Stale entries accumulate until cleanup operations
            cache.insert("temp1", "value1");
            cache.insert("temp2", "value2");
            cache.insert("keep", "value_keep");

            // Manually remove items to create stale entries
            cache.remove_from_cache_only(&"temp1");
            cache.remove_from_cache_only(&"temp2");

            // Stale entries remain in insertion_order
            assert_eq!(cache.len(), 1); // Only "keep" in HashMap
            assert_eq!(cache.insertion_order_len(), 3); // All 3 in insertion_order

            // Verify operations work correctly despite stale entries
            assert_eq!(cache.peek_oldest(), Some((&"keep", &"value_keep")));
            assert_eq!(cache.age_rank(&"keep"), Some(0));
            assert!(!cache.contains(&"temp1"));
            assert!(!cache.contains(&"temp2"));
            assert!(cache.contains(&"keep"));

            // Test 2: Lazy cleanup during pop_oldest
            cache.insert("new1", "value_new1");
            cache.insert("new2", "value_new2");

            // Now we have: stale("temp1"), stale("temp2"), "keep", "new1", "new2"
            assert_eq!(cache.len(), 3);
            assert_eq!(cache.insertion_order_len(), 5);

            // pop_oldest should skip stale entries and pop "keep"
            assert_eq!(cache.pop_oldest(), Some(("keep", "value_keep")));

            // After pop_oldest, some stale entries should be cleaned up
            assert_eq!(cache.len(), 2);
            // insertion_order length depends on how many stale entries were cleaned

            // Test 3: Lazy cleanup during eviction
            cache.insert("trigger_eviction", "value_trigger");

            // This should trigger eviction, which should skip any remaining stale entries
            assert_eq!(cache.len(), 3);
            assert!(cache.contains(&"new1"));
            assert!(cache.contains(&"new2"));
            assert!(cache.contains(&"trigger_eviction"));

            // Test 4: Multiple consecutive stale entries
            cache.clear();
            cache.insert("stale1", "v1");
            cache.insert("stale2", "v2");
            cache.insert("stale3", "v3");
            cache.insert("valid", "valid_value");

            // Remove the first three to create consecutive stale entries
            cache.remove_from_cache_only(&"stale1");
            cache.remove_from_cache_only(&"stale2");
            cache.remove_from_cache_only(&"stale3");

            assert_eq!(cache.len(), 1); // Only "valid" in HashMap
            // insertion_order might have any length >= 1 depending on cleanup behavior

            // Operations should skip all stale entries and work with "valid"
            assert_eq!(cache.peek_oldest(), Some((&"valid", &"valid_value")));
            assert_eq!(cache.pop_oldest(), Some(("valid", "valid_value")));
            assert_eq!(cache.len(), 0);
            assert_eq!(cache.peek_oldest(), None);

            // Cache should be functional after stale entry handling
            cache.insert("new_after_stale", "new_value");
            assert_eq!(cache.len(), 1);
            assert!(cache.contains(&"new_after_stale"));
        }

        #[test]
        fn test_stale_entry_cleanup_during_operations() {
            let mut cache = FIFOCache::new(4);

            // Setup: Create cache with mix of valid and future stale entries
            cache.insert("will_be_stale1", "stale1");
            cache.insert("will_be_stale2", "stale2");
            cache.insert("valid1", "value1");
            cache.insert("valid2", "value2");

            // Create stale entries
            cache.remove_from_cache_only(&"will_be_stale1");
            cache.remove_from_cache_only(&"will_be_stale2");

            assert_eq!(cache.len(), 2); // Only valid entries in HashMap
            assert_eq!(cache.insertion_order_len(), 4); // All entries in insertion_order

            // Test 1: pop_oldest cleans up stale entries as it encounters them
            assert_eq!(cache.pop_oldest(), Some(("valid1", "value1")));

            // Should have cleaned up stale entries encountered during traversal
            assert_eq!(cache.len(), 1);
            // insertion_order may be smaller after cleanup, depending on which stale entries were encountered
            // The exact behavior depends on the order of traversal and which stale entries are cleaned up

            // Test 2: pop_oldest_batch cleans up stale entries
            cache.insert("new1", "new_value1");
            cache.insert("new2", "new_value2");
            cache.insert("new3", "new_value3");

            // Create more stale entries
            cache.remove_from_cache_only(&"new1");
            cache.remove_from_cache_only(&"new3");

            let batch = cache.pop_oldest_batch(2);
            // Should get valid2 and new2 (skipping stale new1 and new3)
            assert_eq!(batch.len(), 2);
            assert!(batch.contains(&("valid2", "value2")));
            assert!(batch.contains(&("new2", "new_value2")));

            // Test 3: age_rank doesn't modify structure but handles stale entries
            cache.insert("test1", "test_value1");
            cache.insert("test2", "test_value2");
            cache.insert("test3", "test_value3");

            cache.remove_from_cache_only(&"test2"); // Make test2 stale

            // age_rank should return correct ranks despite stale entry
            assert_eq!(cache.age_rank(&"test1"), Some(0)); // First valid
            assert_eq!(cache.age_rank(&"test2"), None); // Stale
            assert_eq!(cache.age_rank(&"test3"), Some(1)); // Second valid

            // Test 4: peek_oldest doesn't modify structure but finds valid entry
            cache.remove_from_cache_only(&"test1"); // Make test1 also stale

            // peek_oldest should find test3 despite two stale entries before it
            assert_eq!(cache.peek_oldest(), Some((&"test3", &"test_value3")));

            // The structure should remain unchanged after peek
            assert_eq!(cache.len(), 1); // Only test3 is valid

            // Test 5: Eviction during insertion handles stale entries
            cache.insert("fill1", "f1");
            cache.insert("fill2", "f2");
            cache.insert("fill3", "f3");
            // Now at capacity (4): test3, fill1, fill2, fill3

            // Create stale entries
            cache.remove_from_cache_only(&"fill1");
            cache.remove_from_cache_only(&"fill2");

            // Now we have test3, fill3 in HashMap (2 valid items)
            assert_eq!(cache.len(), 2);

            // Add items to get back to capacity so the next insert will trigger eviction
            cache.insert("temp1", "t1");
            cache.insert("temp2", "t2");
            assert_eq!(cache.len(), 4); // At capacity

            // This insertion should trigger eviction
            cache.insert("trigger", "trigger_value");

            // Should have proper capacity and valid operations
            assert_eq!(cache.len(), 4); // Still at capacity
            assert!(cache.contains(&"trigger"));

            // The cache should still function correctly regardless of internal stale entry cleanup
            cache.insert("final", "final_value");
            assert_eq!(cache.len(), 4); // Should remain at capacity
            assert!(cache.contains(&"final"));

            // Verify the cache maintains proper FIFO behavior
            assert_eq!(cache.capacity(), 4);

            // Test that operations still work normally after stale entry handling
            let oldest = cache.peek_oldest();
            assert!(oldest.is_some()); // Should have valid oldest entry

            let popped = cache.pop_oldest();
            assert!(popped.is_some()); // Should be able to pop
            assert_eq!(cache.len(), 3);
        }
    }
}
