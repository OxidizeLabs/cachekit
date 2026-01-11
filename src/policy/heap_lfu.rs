//! # Heap-Based LFU Cache Implementation
//!
//! This module provides an alternative LFU cache implementation that uses a binary heap for
//! O(log n) eviction operations instead of the O(n) scanning approach used by the standard
//! `LFUCache`.
//!
//! ## Architecture
//!
//! ```text
//!   ┌──────────────────────────────────────────────────────────────────────────┐
//!   │                        HeapLFUCache<K, V>                                │
//!   │                                                                          │
//!   │   ┌────────────────────────────────────────────────────────────────────┐ │
//!   │   │  HashMapStore<K, V>                                               │ │
//!   │   │                                                                    │ │
//!   │   │  ┌─────────┬────────────────────────────────────────────────────┐  │ │
//!   │   │  │   Key   │  Arc<V>                                            │  │ │
//!   │   │  ├─────────┼────────────────────────────────────────────────────┤  │ │
//!   │   │  │ page_1  │  data_1                                            │  │ │
//!   │   │  │ page_2  │  data_2                                            │  │ │
//!   │   │  │ page_3  │  data_3                                            │  │ │
//!   │   │  └─────────┴────────────────────────────────────────────────────┘  │ │
//!   │   └────────────────────────────────────────────────────────────────────┘ │
//!   │                                                                          │
//!   │   ┌────────────────────────────────────────────────────────────────────┐ │
//!   │   │  frequencies: HashMap<K, u64>                                      │ │
//!   │   │                                                                    │ │
//!   │   │  ┌─────────┬──────────┐                                            │ │
//!   │   │  │   Key   │  Freq    │                                            │ │
//!   │   │  ├─────────┼──────────┤                                            │ │
//!   │   │  │ page_1  │  15      │                                            │ │
//!   │   │  │ page_2  │   3      │                                            │ │
//!   │   │  │ page_3  │   7      │                                            │ │
//!   │   │  └─────────┴──────────┘                                            │ │
//!   │   └────────────────────────────────────────────────────────────────────┘ │
//!   │                                                                          │
//!   │   ┌────────────────────────────────────────────────────────────────────┐ │
//!   │   │  freq_heap: BinaryHeap<Reverse<(u64, K)>>  (Min-Heap)              │ │
//!   │   │                                                                    │ │
//!   │   │  ┌─────────────────────────────────────────────────────────────┐   │ │
//!   │   │  │                        (3, page_2)  ← top (min freq)        │   │ │
//!   │   │  │                       /            \                        │   │ │
//!   │   │  │               (7, page_3)      (15, page_1)                 │   │ │
//!   │   │  │                                                             │   │ │
//!   │   │  │  Note: May contain stale entries with outdated frequencies  │   │ │
//!   │   │  └─────────────────────────────────────────────────────────────┘   │ │
//!   │   └────────────────────────────────────────────────────────────────────┘ │
//!   │                                                                          │
//!   │   capacity: usize                                                        │
//!   └──────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Stale Entry Handling
//!
//! ```text
//!   Problem: BinaryHeap doesn't support efficient arbitrary element updates
//!   Solution: Lazy invalidation with freshness check during eviction
//!
//!   ═══════════════════════════════════════════════════════════════════════════
//!
//!   Initial state:
//!     frequencies: { A: 1 }
//!     heap: [ (1, A) ]
//!
//!   get(&A):  Increment frequency
//!     frequencies: { A: 2 }           ← Updated to 2
//!     heap: [ (1, A), (2, A) ]        ← New entry added, old becomes STALE
//!                 ↑
//!               stale
//!
//!   get(&A) again:
//!     frequencies: { A: 3 }
//!     heap: [ (1, A), (2, A), (3, A) ]
//!                 ↑       ↑
//!              stale   stale
//!
//!   pop_lfu():
//!     1. Pop (1, A) from heap
//!     2. Check: frequencies[A] == 1?  NO, it's 3
//!     3. Entry is STALE → discard, try next
//!     4. Pop (2, A) from heap
//!     5. Check: frequencies[A] == 2?  NO, it's 3
//!     6. Entry is STALE → discard, try next
//!     7. Pop (3, A) from heap
//!     8. Check: frequencies[A] == 3?  YES, valid!
//!     9. Return (A, value)
//!
//!   ═══════════════════════════════════════════════════════════════════════════
//! ```
//!
//! ### Bounded Heap Cleanup
//!
//! To keep heap growth bounded under heavy access churn, the heap is rebuilt from the
//! authoritative `frequencies` map once it grows beyond a fixed multiple of live entries.
//! This drops stale entries in bulk while preserving LFU ordering.
//!
//! ## Comparison: Standard LFU vs. Heap LFU
//!
//! ```text
//!   Standard LFUCache:
//!   ┌────────────────────────────────────────────────────────────────────────┐
//!   │  HashMap<K, (V, usize)>                                                │
//!   │                                                                        │
//!   │  insert/get: O(1)                                                      │
//!   │  pop_lfu:    O(n)  ← Must scan ALL entries to find minimum             │
//!   │                                                                        │
//!   │  Memory: 1 HashMap                                                     │
//!   └────────────────────────────────────────────────────────────────────────┘
//!
//!   HeapLFUCache:
//!   ┌────────────────────────────────────────────────────────────────────────┐
//!   │  HashMapStore<K, V> + HashMap<K, u64> + BinaryHeap<(u64, K)>           │
//!   │                                                                        │
//!   │  insert/get: O(log n)  ← Heap operations                               │
//!   │  pop_lfu:    O(log n)  ← Pop from heap (amortized, skipping stale)     │
//!   │                                                                        │
//!   │  Memory: 3 data structures + stale entries                             │
//!   └────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Key Components
//!
//! | Component      | Type                              | Purpose                    |
//! |----------------|-----------------------------------|----------------------------|
//! | `store`        | `HashMapStore<K, V>`             | Value storage, O(1) lookup |
//! | `frequencies`  | `HashMap<K, u64>`                 | Current frequency tracking |
//! | `freq_heap`    | `BinaryHeap<Reverse<(u64, K)>>`   | Min-heap for LFU lookup    |
//! | `capacity`     | `usize`                           | Maximum entries            |
//!
//! ## Core Operations
//!
//! | Method           | Complexity | Description                              |
//! |------------------|------------|------------------------------------------|
//! | `new(capacity)`  | O(1)       | Create cache with given capacity         |
//! | `insert(k, v)`   | O(log n)   | Insert `Arc<V>`, may trigger eviction    |
//! | `get(&k)`        | O(log n)   | Get value, increments frequency          |
//! | `contains(&k)`   | O(1)       | Check if key exists                      |
//! | `remove(&k)`     | O(1)       | Remove entry (lazy heap cleanup)         |
//! | `len()`          | O(1)       | Current number of entries                |
//! | `capacity()`     | O(1)       | Maximum capacity                         |
//! | `clear()`        | O(n)       | Remove all entries                       |
//!
//! ## LFU-Specific Operations
//!
//! | Method                   | Complexity | Description                       |
//! |--------------------------|------------|-----------------------------------|
//! | `pop_lfu()`              | O(log n)*  | Remove and return LFU item        |
//! | `peek_lfu()`             | O(n)       | Peek at LFU (falls back to scan)  |
//! | `frequency(&k)`          | O(1)       | Get frequency count for key       |
//! | `reset_frequency(&k)`    | O(log n)   | Reset frequency to 1              |
//! | `increment_frequency(&k)`| O(log n)   | Manually increment frequency      |
//!
//! \* Amortized, may skip stale entries
//!
//! ## Performance Trade-offs
//!
//! | Aspect           | Standard LFU       | Heap LFU                    |
//! |------------------|--------------------|-----------------------------|
//! | `get/insert`     | O(1)               | O(log n)                    |
//! | `pop_lfu`        | O(n)               | O(log n) amortized          |
//! | Memory           | Store + maps       | 3 data structures + stale   |
//! | Constant factors | Low                | Higher                      |
//! | Predictability   | O(n) worst case    | Consistent O(log n)         |
//! | Lock time        | Longer during evict| Shorter, more consistent    |
//!
//! ## When to Use
//!
//! **Use HeapLFUCache when:**
//! - Eviction operations are frequent (>10% of operations)
//! - Consistent, predictable latency is critical
//! - Cache sizes are large (>1000 items)
//! - High-throughput, low-latency workloads
//!
//! **Use standard LFUCache when:**
//! - Memory usage is critical
//! - Evictions are rare compared to gets
//! - Cache sizes are small (<100 items)
//! - Simple, minimal overhead is preferred
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! use crate::storage::disk::async_disk::cache::heap_lfu::HeapLFUCache;
//! use std::sync::Arc;
//! use crate::storage::disk::async_disk::cache::cache_traits::{
//!     CoreCache, MutableCache, LFUCacheTrait,
//! };
//!
//! // Create cache
//! let mut cache: HeapLFUCache<String, i32> = HeapLFUCache::new(100);
//!
//! // Insert items (frequency starts at 1)
//! cache.insert("key1".to_string(), Arc::new(100));
//! cache.insert("key2".to_string(), Arc::new(200));
//!
//! // Access increments frequency (O(log n) for heap update)
//! cache.get(&"key1".to_string()); // freq: 1 → 2
//! cache.get(&"key1".to_string()); // freq: 2 → 3
//!
//! assert_eq!(cache.frequency(&"key1".to_string()), Some(3));
//! assert_eq!(cache.frequency(&"key2".to_string()), Some(1));
//!
//! // Evict LFU item (O(log n) amortized)
//! if let Some((key, value)) = cache.pop_lfu() {
//!     println!("Evicted: {} = {}", key, value.as_ref());
//! }
//!
//! // Manual frequency control
//! cache.increment_frequency(&"key2".to_string());
//! cache.reset_frequency(&"key1".to_string());
//!
//! // Remove (O(1), lazy heap cleanup)
//! cache.remove(&"key1".to_string());
//! ```
//!
//! ## Type Constraints
//!
//! ```text
//!   K: Eq + Hash + Clone + Ord
//!        │    │      │      │
//!        │    │      │      └── Required for BinaryHeap ordering
//!        │    │      └───────── Required for heap entry cloning
//!        │    └──────────────── Required for HashMap
//!        └───────────────────── Required for HashMap
//!
//!   V: (no constraints, values are stored as `Arc<V>`)
//! ```
//!
//! ## Thread Safety
//!
//! - `HeapLFUCache` is **NOT thread-safe**
//! - Wrap in `Arc<Mutex<HeapLFUCache>>` for concurrent access
//! - Shorter lock times than standard LFU due to O(log n) eviction
//!
//! ## Implementation Notes
//!
//! - **Stale entries**: Accumulate in heap, cleaned lazily during `pop_lfu()`
//! - **Bounded rebuilds**: Heap is rebuilt when size exceeds `MAX_HEAP_FACTOR * live_entries`
//! - **peek_lfu()**: Falls back to O(n) scan (avoiding heap borrow issues)
//! - **Memory overhead**: ~3x standard LFU due to three data structures
//! - **Reverse wrapper**: Converts max-heap to min-heap for LFU semantics

use crate::store::hashmap::HashMapStore;
use crate::store::traits::{StoreCore, StoreMut};
use crate::traits::{CoreCache, LFUCacheTrait, MutableCache};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::hash::Hash;
use std::sync::Arc;

/// Heap-based LFU Cache with O(log n) eviction.
///
/// Uses a binary min-heap for efficient LFU item identification.
/// Values are stored as `Arc<V>` to avoid cloning on eviction.
/// See module-level documentation for details.
#[derive(Debug)]
pub struct HeapLFUCache<K, V>
where
    K: Eq + Hash + Clone + Ord,
{
    store: HashMapStore<K, V>,
    frequencies: HashMap<K, u64>,
    // Min-heap: smallest frequency first
    // Reverse wrapper converts max-heap to min-heap
    freq_heap: BinaryHeap<Reverse<(u64, K)>>,
}

impl<K, V> HeapLFUCache<K, V>
where
    K: Eq + Hash + Clone + Ord,
{
    const MAX_HEAP_FACTOR: usize = 4;

    /// Creates a new HeapLFUCache with the specified capacity.
    ///
    /// # Arguments
    /// * `capacity` - Maximum number of items the cache can hold
    ///
    /// # Examples
    /// ```rust,no_run
    /// use cachekit::policy::heap_lfu::HeapLFUCache;
    ///
    /// let cache: HeapLFUCache<String, i32> = HeapLFUCache::new(100);
    /// assert_eq!(cache.capacity(), 100);
    /// assert_eq!(cache.len(), 0);
    /// ```
    pub fn new(capacity: usize) -> Self {
        HeapLFUCache {
            store: HashMapStore::new(capacity),
            frequencies: HashMap::with_capacity(capacity),
            freq_heap: BinaryHeap::with_capacity(capacity),
        }
    }

    /// Returns the maximum capacity of the cache.
    pub fn capacity(&self) -> usize {
        self.store.capacity()
    }

    /// Returns the current number of items in the cache.
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Returns true if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Checks if the cache contains the specified key.
    ///
    /// This operation is O(1) and does not affect access frequencies.
    ///
    /// # Arguments
    /// * `key` - The key to check for
    ///
    /// # Returns
    /// `true` if the key exists in the cache, `false` otherwise
    pub fn contains(&self, key: &K) -> bool {
        self.store.contains(key)
    }

    /// Gets the current access frequency for a key.
    ///
    /// # Arguments
    /// * `key` - The key to get frequency for
    ///
    /// # Returns
    /// `Some(frequency)` if the key exists, `None` otherwise
    pub fn frequency(&self, key: &K) -> Option<u64> {
        self.frequencies.get(key).copied()
    }

    /// Clears all items from the cache.
    pub fn clear(&mut self) {
        self.store.clear();
        self.frequencies.clear();
        self.freq_heap.clear();
    }

    /// Private helper to add a frequency entry to the heap.
    fn add_to_heap(&mut self, key: &K, frequency: u64) {
        self.freq_heap.push(Reverse((frequency, key.clone())));
        self.maybe_rebuild_heap();
    }

    /// Bound the heap size by rebuilding from the authoritative frequencies map.
    fn maybe_rebuild_heap(&mut self) {
        let live_entries = self.store.len().max(1);
        let max_heap_len = live_entries.saturating_mul(Self::MAX_HEAP_FACTOR);

        if self.freq_heap.len() <= max_heap_len {
            return;
        }

        self.freq_heap.clear();
        self.freq_heap.reserve(self.frequencies.len());
        for (key, freq) in &self.frequencies {
            self.freq_heap.push(Reverse((*freq, key.clone())));
        }
    }

    /// Private helper to remove stale entries from the front of the heap.
    /// Returns the current minimum frequency key, or None if cache is empty.
    fn pop_lfu_internal(&mut self) -> Option<(K, u64)> {
        let mut stale_pops = 0usize;
        while let Some(Reverse((heap_freq, key))) = self.freq_heap.peek() {
            if let Some(&current_freq) = self.frequencies.get(key)
                && *heap_freq == current_freq
            {
                // This is a valid (non-stale) entry
                let Reverse((freq, key)) = self.freq_heap.pop().unwrap();
                return Some((key, freq));
            }

            // This entry is stale (key doesn't exist or frequency changed)
            self.freq_heap.pop();
            stale_pops += 1;
            if stale_pops >= self.store.len().max(1) {
                self.maybe_rebuild_heap();
                stale_pops = 0;
            }
        }

        None
    }

    /// Private helper to ensure cache doesn't exceed capacity.
    /// Evicts the least frequently used item if necessary.
    fn ensure_capacity(&mut self) -> Option<(K, Arc<V>)> {
        if self.store.len() >= self.store.capacity() {
            self.pop_lfu()
        } else {
            None
        }
    }
}

// Implementation of CoreCache trait
impl<K, V> CoreCache<K, Arc<V>> for HeapLFUCache<K, V>
where
    K: Eq + Hash + Clone + Ord,
{
    fn insert(&mut self, key: K, value: Arc<V>) -> Option<Arc<V>> {
        // If key already exists, just update the value (don't change frequency)
        if self.store.contains(&key) {
            return self.store.try_insert(key, value).ok().flatten();
        }

        // Ensure we have capacity (may evict LFU item)
        self.ensure_capacity();

        // Insert new item with frequency 1
        if self.store.try_insert(key.clone(), value).is_err() {
            return None;
        }
        self.frequencies.insert(key.clone(), 1);
        self.add_to_heap(&key, 1);

        None
    }

    fn get(&mut self, key: &K) -> Option<&Arc<V>> {
        if self.store.contains(key) {
            // Increment frequency
            let new_freq = self.frequencies.get_mut(key).map(|f| {
                *f += 1;
                *f
            })?;

            // Add new frequency entry to heap (old entry becomes stale)
            self.add_to_heap(key, new_freq);

            self.store.get_ref(key)
        } else {
            None
        }
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
        self.store.clear();
        self.frequencies.clear();
        self.freq_heap.clear();
    }
}

// Implementation of MutableCache trait for arbitrary key removal
impl<K, V> MutableCache<K, Arc<V>> for HeapLFUCache<K, V>
where
    K: Eq + Hash + Clone + Ord,
{
    fn remove(&mut self, key: &K) -> Option<Arc<V>> {
        // Remove from store and frequencies maps
        let value = self.store.remove(key);
        let had_frequency = self.frequencies.remove(key).is_some();

        // Note: We don't remove from heap immediately (lazy removal)
        // Stale entries will be filtered out during pop_lfu operations

        if value.is_some() || had_frequency {
            self.maybe_rebuild_heap();
        }

        value
    }
}

// Implementation of LFUCacheTrait for specialized LFU operations
impl<K, V> LFUCacheTrait<K, Arc<V>> for HeapLFUCache<K, V>
where
    K: Eq + Hash + Clone + Ord,
{
    fn pop_lfu(&mut self) -> Option<(K, Arc<V>)> {
        // Find the key with minimum frequency (handling stale entries)
        let (lfu_key, _freq) = self.pop_lfu_internal()?;

        // Remove from all data structures
        let value = self.store.remove(&lfu_key)?;
        self.frequencies.remove(&lfu_key);
        self.store.record_eviction();

        Some((lfu_key, value))
    }

    fn peek_lfu(&self) -> Option<(&K, &Arc<V>)> {
        // This is more expensive for heap-based approach since we need to
        // scan through potential stale entries. For better performance,
        // consider avoiding this operation if possible.

        // Find the key with minimum frequency by scanning the frequencies map
        // This is O(n) but avoids the borrowing issues with heap cloning
        if self.frequencies.is_empty() {
            return None;
        }

        let min_freq = *self.frequencies.values().min()?;

        // Find a key with the minimum frequency
        for (key, &freq) in &self.frequencies {
            if freq == min_freq {
                return self.store.peek_ref(key).map(|value| (key, value));
            }
        }

        None
    }

    fn frequency(&self, key: &K) -> Option<u64> {
        self.frequencies.get(key).copied()
    }

    fn increment_frequency(&mut self, key: &K) -> Option<u64> {
        if let Some(freq) = self.frequencies.get_mut(key) {
            *freq += 1;
            let new_freq = *freq;
            self.add_to_heap(key, new_freq);
            Some(new_freq)
        } else {
            None
        }
    }

    fn reset_frequency(&mut self, key: &K) -> Option<u64> {
        if let Some(freq) = self.frequencies.get_mut(key) {
            let old_freq = *freq;
            *freq = 1;
            self.add_to_heap(key, 1);
            Some(old_freq)
        } else {
            None
        }
    }
}

// ==============================================
// HEAP LFU CACHE TESTS
// ==============================================

#[cfg(test)]
mod heap_lfu_tests {
    use super::*;
    use crate::policy::lfu::LFUCache;

    #[test]
    fn test_heap_lfu_basic_operations() {
        let mut cache: HeapLFUCache<String, i32> = HeapLFUCache::new(3);

        // Test basic insertion and retrieval
        assert_eq!(cache.insert("key1".to_string(), Arc::new(100)), None);
        assert_eq!(cache.insert("key2".to_string(), Arc::new(200)), None);
        assert_eq!(cache.insert("key3".to_string(), Arc::new(300)), None);

        assert_eq!(cache.len(), 3);
        assert_eq!(cache.capacity(), 3);

        // Test retrieval and frequency tracking
        assert_eq!(cache.get(&"key1".to_string()).map(Arc::as_ref), Some(&100));
        assert_eq!(cache.frequency(&"key1".to_string()), Some(2)); // 1 + 1 from get

        assert_eq!(cache.get(&"key2".to_string()).map(Arc::as_ref), Some(&200));
        assert_eq!(cache.get(&"key2".to_string()).map(Arc::as_ref), Some(&200)); // Access again
        assert_eq!(cache.frequency(&"key2".to_string()), Some(3)); // 1 + 2 from gets

        // Test contains
        assert!(cache.contains(&"key1".to_string()));
        assert!(!cache.contains(&"nonexistent".to_string()));
    }

    #[test]
    fn test_heap_lfu_eviction_order() {
        let mut cache: HeapLFUCache<String, i32> = HeapLFUCache::new(3);

        // Fill cache to capacity
        cache.insert("key1".to_string(), Arc::new(100));
        cache.insert("key2".to_string(), Arc::new(200));
        cache.insert("key3".to_string(), Arc::new(300));

        // Create different access patterns to establish frequency order
        // key1: frequency = 1 (no additional accesses)
        // key2: frequency = 3 (2 additional accesses)
        // key3: frequency = 2 (1 additional access)
        cache.get(&"key2".to_string()); // key2 freq = 2
        cache.get(&"key2".to_string()); // key2 freq = 3
        cache.get(&"key3".to_string()); // key3 freq = 2

        // Verify frequencies before eviction
        assert_eq!(cache.frequency(&"key1".to_string()), Some(1)); // LFU
        assert_eq!(cache.frequency(&"key2".to_string()), Some(3)); // MFU
        assert_eq!(cache.frequency(&"key3".to_string()), Some(2)); // Middle

        // Insert new item - should evict key1 (LFU)
        cache.insert("key4".to_string(), Arc::new(400));

        // Verify key1 was evicted (LFU)
        assert!(!cache.contains(&"key1".to_string()));
        assert_eq!(cache.get(&"key1".to_string()).map(Arc::as_ref), None);

        // Verify other keys still exist
        assert!(cache.contains(&"key2".to_string()));
        assert!(cache.contains(&"key3".to_string()));
        assert!(cache.contains(&"key4".to_string()));

        // Verify cache size
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn test_heap_lfu_pop_lfu() {
        let mut cache: HeapLFUCache<String, i32> = HeapLFUCache::new(3);

        // Insert items with different frequencies
        cache.insert("low".to_string(), Arc::new(1));
        cache.insert("med".to_string(), Arc::new(2));
        cache.insert("high".to_string(), Arc::new(3));

        // Create frequency differences
        cache.get(&"med".to_string()); // med freq = 2
        cache.get(&"high".to_string()); // high freq = 2
        cache.get(&"high".to_string()); // high freq = 3

        // Expected frequencies: low=1, med=2, high=3
        assert_eq!(cache.frequency(&"low".to_string()), Some(1));
        assert_eq!(cache.frequency(&"med".to_string()), Some(2));
        assert_eq!(cache.frequency(&"high".to_string()), Some(3));

        // Pop LFU should return "low"
        let (key, value) = cache.pop_lfu().unwrap();
        assert_eq!(key, "low".to_string());
        assert_eq!(*value, 1);
        assert_eq!(cache.len(), 2);

        // Pop LFU should now return "med"
        let (key, value) = cache.pop_lfu().unwrap();
        assert_eq!(key, "med".to_string());
        assert_eq!(*value, 2);
        assert_eq!(cache.len(), 1);

        // Pop LFU should now return "high"
        let (key, value) = cache.pop_lfu().unwrap();
        assert_eq!(key, "high".to_string());
        assert_eq!(*value, 3);
        assert_eq!(cache.len(), 0);

        // Pop LFU on empty cache should return None
        assert_eq!(cache.pop_lfu(), None);
    }

    #[test]
    fn test_heap_lfu_stale_entry_handling() {
        let mut cache: HeapLFUCache<i32, i32> = HeapLFUCache::new(3);

        // Insert items
        cache.insert(1, Arc::new(10));
        cache.insert(2, Arc::new(20));
        cache.insert(3, Arc::new(30));

        // Access to create heap entries
        cache.get(&1); // freq = 2
        cache.get(&1); // freq = 3
        cache.get(&2); // freq = 2

        // Remove one item (creates stale heap entries)
        cache.remove(&1);

        // Insert new item to trigger eviction
        cache.insert(4, Arc::new(40));

        // Should still work correctly despite stale entries
        assert!(!cache.contains(&1));
        assert!(cache.contains(&2));
        assert!(cache.contains(&3));
        assert!(cache.contains(&4));
        assert_eq!(cache.len(), 3);

        // Pop LFU should correctly skip stale entries and return valid item
        let (key, _) = cache.pop_lfu().unwrap();
        assert!(key == 3 || key == 4); // Both have frequency 1
    }

    #[test]
    fn test_remove_clears_stale_frequency_entries() {
        let mut cache: HeapLFUCache<String, i32> = HeapLFUCache::new(2);

        cache.insert("key1".to_string(), Arc::new(10));
        cache.insert("key2".to_string(), Arc::new(20));

        for _ in 0..10 {
            cache.increment_frequency(&"key1".to_string());
        }

        let _ = cache.store.remove(&"key1".to_string());
        assert!(cache.frequency(&"key1".to_string()).is_some());

        cache.remove(&"key1".to_string());
        assert!(cache.frequency(&"key1".to_string()).is_none());

        let has_key1_in_heap = cache
            .freq_heap
            .iter()
            .any(|Reverse((_, key))| key == "key1");
        assert!(!has_key1_in_heap);
    }

    #[test]
    fn test_pop_lfu_internal_rebuilds_after_stale_pops() {
        let mut cache: HeapLFUCache<String, i32> = HeapLFUCache::new(2);

        cache.insert("key1".to_string(), Arc::new(10));
        cache.insert("key2".to_string(), Arc::new(20));

        for _ in 0..10 {
            cache.increment_frequency(&"key1".to_string());
        }

        for _ in 0..4 {
            cache.increment_frequency(&"key2".to_string());
        }

        let heap_len_before = cache.freq_heap.len();
        let (lfu_key, lfu_freq) = cache.pop_lfu_internal().unwrap();

        assert_eq!(lfu_key, "key2".to_string());
        assert_eq!(lfu_freq, 5);
        assert!(cache.freq_heap.len() < heap_len_before);
        assert_eq!(cache.freq_heap.len(), 1);
    }

    #[test]
    fn test_heap_lfu_performance_comparison() {
        use std::time::Instant;

        // Test performance comparison between standard LFU and HeapLFU
        let cache_size = 100;

        // Test standard LFU cache
        let mut std_cache = LFUCache::new(cache_size);

        // Fill cache
        for i in 0..cache_size {
            std_cache.insert(i, Arc::new(i * 10));
        }

        // Time pop_lfu operations on standard cache
        let start = Instant::now();
        for _ in 0..10 {
            if let Some((key, value)) = std_cache.pop_lfu() {
                std_cache.insert(key + cache_size, value); // Re-insert with different key
            }
        }
        let std_duration = start.elapsed();

        // Test heap-based LFU cache
        let mut heap_cache: HeapLFUCache<usize, usize> = HeapLFUCache::new(cache_size);

        // Fill cache
        for i in 0..cache_size {
            heap_cache.insert(i, Arc::new(i * 10));
        }

        // Time pop_lfu operations on heap cache
        let start = Instant::now();
        for _ in 0..10 {
            if let Some((key, value)) = heap_cache.pop_lfu() {
                heap_cache.insert(key + cache_size, value); // Re-insert with different key
            }
        }
        let heap_duration = start.elapsed();

        println!("Performance Comparison:");
        println!("  Standard LFU (O(n)): {:?}", std_duration);
        println!("  Heap LFU (O(log n)): {:?}", heap_duration);

        // For larger cache sizes, heap-based should be faster for eviction-heavy workloads
        // Note: For small caches, standard LFU might be faster due to lower constant factors

        // Verify both caches work correctly
        assert_eq!(std_cache.len(), cache_size);
        assert_eq!(heap_cache.len(), cache_size);
    }
}
