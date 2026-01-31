//! # Heap-Based MFU Cache Implementation
//!
//! This module provides a Most Frequently Used (MFU) cache implementation that uses a binary heap
//! to track and evict the entry with the highest access frequency.
//!
//! ## Architecture
//!
//! ```text
//!   ┌──────────────────────────────────────────────────────────────────────────┐
//!   │                        HeapMfuCache<K, V>                                │
//!   │                                                                          │
//!   │   ┌────────────────────────────────────────────────────────────────────┐ │
//!   │   │  HashMapStore<K, V>                                                │ │
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
//!   │   │  freq_heap: BinaryHeap<(u64, K)>  (Max-Heap)                       │ │
//!   │   │                                                                    │ │
//!   │   │  ┌─────────────────────────────────────────────────────────────┐   │ │
//!   │   │  │                        (15, page_1)  ← top (max freq)       │   │ │
//!   │   │  │                       /            \                        │   │ │
//!   │   │  │               (7, page_3)      (3, page_2)                  │   │ │
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
//!     frequencies: { A: 3 }
//!     heap: [ (3, A) ]
//!
//!   get(&A):  Increment frequency
//!     frequencies: { A: 4 }           ← Updated to 4
//!     heap: [ (4, A), (3, A) ]        ← New entry added, old becomes STALE
//!                         ↑
//!                       stale
//!
//!   get(&A) again:
//!     frequencies: { A: 5 }
//!     heap: [ (5, A), (4, A), (3, A) ]
//!                         ↑       ↑
//!                      stale   stale
//!
//!   pop_mfu():
//!     1. Pop (5, A) from heap
//!     2. Check: frequencies[A] == 5?  YES, valid!
//!     3. Return (A, value)
//!
//!   ═══════════════════════════════════════════════════════════════════════════
//! ```
//!
//! ## Comparison: LFU vs. MFU
//!
//! ```text
//!   LFU (Least Frequently Used):
//!   ┌─────────────────────────────────────────────────────────────────────────┐
//!   │  Uses MIN-HEAP: Evicts entry with LOWEST frequency                      │
//!   │  ┌──────────────────────────────────────────────────────────────────┐   │
//!   │  │  heap: [ (1, A), (2, B), (5, C) ]                                │   │
//!   │  │           ↑                                                      │   │
//!   │  │         evict this (min freq)                                    │   │
//!   │  └──────────────────────────────────────────────────────────────────┘   │
//!   │  Good for: General caching (keep frequently used items)                 │
//!   └─────────────────────────────────────────────────────────────────────────┘
//!
//!   MFU (Most Frequently Used):
//!   ┌─────────────────────────────────────────────────────────────────────────┐
//!   │  Uses MAX-HEAP: Evicts entry with HIGHEST frequency                     │
//!   │  ┌──────────────────────────────────────────────────────────────────┐   │
//!   │  │  heap: [ (5, C), (2, B), (1, A) ]                                │   │
//!   │  │           ↑                                                      │   │
//!   │  │         evict this (max freq)                                    │   │
//!   │  └──────────────────────────────────────────────────────────────────┘   │
//!   │  Good for: Niche cases where "most frequent" = "least needed next"      │
//!   └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Example Usage
//!
//! ```
//! use cachekit::policy::mfu::MfuCore;
//!
//! // Create MFU cache with capacity 3
//! let mut cache = MfuCore::new(3);
//!
//! // Insert and access items
//! cache.insert(1, 100);
//! cache.insert(2, 200);
//! cache.insert(3, 300);
//!
//! // Access item 1 multiple times (highest frequency)
//! for _ in 0..10 {
//!     cache.get(&1);
//! }
//!
//! // Access item 2 a few times
//! cache.get(&2);
//! cache.get(&2);
//!
//! // When cache is full, item 1 (highest frequency) will be evicted!
//! cache.insert(4, 400);
//!
//! assert!(!cache.contains(&1)); // Item 1 was most frequently used, evicted
//! assert!(cache.contains(&2));  // Item 2 still in cache
//! assert!(cache.contains(&3));  // Item 3 still in cache
//! assert!(cache.contains(&4));  // Item 4 just inserted
//! ```
//!
//! ## Thread Safety
//!
//! - [`MfuCore`]: Not thread-safe, designed for single-threaded use
//! - For concurrent access, wrap in external synchronization
//!
//! ## Implementation Notes
//!
//! - Uses max-heap (BinaryHeap without Reverse wrapper)
//! - Lazy stale entry cleanup during eviction
//! - Periodic heap rebuild to drop stale entries
//! - O(log n) insert/get, amortized O(log n) eviction
//!
//! ## When to Use MFU
//!
//! MFU is counterintuitive for most workloads but useful in specific scenarios:
//!
//! - **Burst detection**: Evict items that had a temporary burst of activity
//! - **Anti-scan**: Items with high frequency might be from a one-time scan
//! - **Baseline comparisons**: Benchmark against optimal policies
//!
//! **⚠️ Warning**: MFU performs poorly for typical workloads with temporal locality.
//! Use LFU, LRU, or S3-FIFO for general-purpose caching.

use crate::traits::CoreCache;
use rustc_hash::FxHashMap;
use std::collections::BinaryHeap;
use std::fmt;
use std::hash::Hash;

/// Max-heap rebuilds when stale entries exceed this factor times live entries.
const HEAP_REBUILD_FACTOR: usize = 3;

/// MFU cache core that evicts the most frequently used entry.
pub struct MfuCore<K, V> {
    map: FxHashMap<K, V>,
    frequencies: FxHashMap<K, u64>,
    freq_heap: BinaryHeap<(u64, K)>, // Max-heap (no Reverse wrapper)
    capacity: usize,
}

impl<K, V> MfuCore<K, V>
where
    K: Clone + Eq + Hash + Ord,
{
    /// Creates a new MFU cache with the specified capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            map: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
            frequencies: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
            freq_heap: BinaryHeap::with_capacity(capacity),
            capacity,
        }
    }

    /// Gets a value by key, incrementing its frequency.
    pub fn get(&mut self, key: &K) -> Option<&V> {
        if self.map.contains_key(key) {
            // Increment frequency
            let freq = self.frequencies.entry(key.clone()).or_insert(0);
            *freq += 1;

            // Push new (freq, key) entry to heap (old entries become stale)
            self.freq_heap.push((*freq, key.clone()));

            // Rebuild heap if too many stale entries accumulated
            if self.freq_heap.len() > self.map.len() * HEAP_REBUILD_FACTOR {
                self.rebuild_heap();
            }

            self.map.get(key)
        } else {
            None
        }
    }

    /// Inserts a key-value pair, evicting the most frequently used entry if at capacity.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        if self.capacity == 0 {
            return Some(value);
        }

        // Update or insert
        let result = if self.map.contains_key(&key) {
            // Update existing entry
            let old_value = self.map.insert(key.clone(), value);
            let freq = self.frequencies.entry(key.clone()).or_insert(0);
            *freq += 1;
            self.freq_heap.push((*freq, key));
            old_value
        } else {
            // Need to evict if at capacity
            while self.map.len() >= self.capacity {
                self.evict_mfu();
            }

            // Insert new entry
            self.map.insert(key.clone(), value);
            self.frequencies.insert(key.clone(), 1);
            self.freq_heap.push((1, key));
            None
        };

        #[cfg(debug_assertions)]
        self.validate_invariants();

        result
    }

    /// Evicts the entry with the highest frequency (MFU).
    fn evict_mfu(&mut self) {
        while let Some((heap_freq, key)) = self.freq_heap.pop() {
            // Check if this heap entry is stale
            if let Some(&current_freq) = self.frequencies.get(&key) {
                if current_freq == heap_freq {
                    // Valid entry, evict it
                    self.map.remove(&key);
                    self.frequencies.remove(&key);

                    #[cfg(debug_assertions)]
                    self.validate_invariants();

                    return;
                }
                // Stale entry, continue to next
            }
        }

        // Heap empty but map not empty? Rebuild and try again
        if !self.map.is_empty() {
            self.rebuild_heap();
            if let Some((_, key)) = self.freq_heap.pop() {
                self.map.remove(&key);
                self.frequencies.remove(&key);

                #[cfg(debug_assertions)]
                self.validate_invariants();
            }
        }
    }

    /// Rebuilds the heap from current frequencies, dropping all stale entries.
    fn rebuild_heap(&mut self) {
        self.freq_heap.clear();
        for (key, &freq) in &self.frequencies {
            self.freq_heap.push((freq, key.clone()));
        }
    }

    /// Returns the number of entries currently in the cache.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns true if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Returns the cache capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Checks if a key exists in the cache without updating frequency.
    pub fn contains(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    /// Removes all entries from the cache.
    pub fn clear(&mut self) {
        self.map.clear();
        self.frequencies.clear();
        self.freq_heap.clear();

        #[cfg(debug_assertions)]
        self.validate_invariants();
    }

    /// Gets the current frequency count for a key.
    pub fn frequency(&self, key: &K) -> Option<u64> {
        self.frequencies.get(key).copied()
    }

    /// Removes and returns the entry with the highest frequency.
    pub fn pop_mfu(&mut self) -> Option<(K, V)> {
        while let Some((heap_freq, key)) = self.freq_heap.pop() {
            if let Some(&current_freq) = self.frequencies.get(&key) {
                if current_freq == heap_freq {
                    // Valid entry
                    if let Some(value) = self.map.remove(&key) {
                        self.frequencies.remove(&key);

                        #[cfg(debug_assertions)]
                        self.validate_invariants();

                        return Some((key, value));
                    }
                }
            }
        }
        None
    }

    /// Peeks at the entry with highest frequency without removing it.
    pub fn peek_mfu(&self) -> Option<(&K, &V)> {
        // Find max frequency
        let mut max_freq = 0u64;
        let mut max_key: Option<&K> = None;

        for (key, &freq) in &self.frequencies {
            if freq > max_freq {
                max_freq = freq;
                max_key = Some(key);
            }
        }

        max_key.and_then(|k| self.map.get(k).map(|v| (k, v)))
    }

    /// Validates internal data structure invariants.
    ///
    /// This method checks that:
    /// - All keys in `map` have corresponding frequency entries
    /// - All keys in `frequencies` exist in `map`
    /// - The sizes of `map` and `frequencies` match
    ///
    /// Only runs when debug assertions are enabled.
    #[cfg(debug_assertions)]
    fn validate_invariants(&self) {
        // Map and frequencies should have same size
        debug_assert_eq!(
            self.map.len(),
            self.frequencies.len(),
            "Map and frequencies have different sizes: map={}, freq={}",
            self.map.len(),
            self.frequencies.len()
        );

        // All keys in map must have frequency entries
        for key in self.map.keys() {
            debug_assert!(
                self.frequencies.contains_key(key),
                "Key in map but not in frequencies"
            );
        }

        // All keys in frequencies must exist in map
        for key in self.frequencies.keys() {
            debug_assert!(
                self.map.contains_key(key),
                "Key in frequencies but not in map"
            );
        }

        // Verify all frequencies are at least 1
        for &freq in self.frequencies.values() {
            debug_assert!(freq >= 1, "Invalid frequency found: {}", freq);
        }

        // Heap can contain stale entries, so we just verify it's bounded
        debug_assert!(
            self.freq_heap.len() <= self.map.len() * (HEAP_REBUILD_FACTOR + 1),
            "Heap size {} exceeds reasonable bounds for map size {}",
            self.freq_heap.len(),
            self.map.len()
        );
    }
}

impl<K, V> CoreCache<K, V> for MfuCore<K, V>
where
    K: Clone + Eq + Hash + Ord,
{
    fn get(&mut self, key: &K) -> Option<&V> {
        if self.map.contains_key(key) {
            // Increment frequency
            let freq = self.frequencies.entry(key.clone()).or_insert(0);
            *freq += 1;

            // Push new (freq, key) entry to heap (old entries become stale)
            self.freq_heap.push((*freq, key.clone()));

            // Rebuild heap if too many stale entries accumulated
            if self.freq_heap.len() > self.map.len() * HEAP_REBUILD_FACTOR {
                self.rebuild_heap();
            }

            self.map.get(key)
        } else {
            None
        }
    }

    fn insert(&mut self, key: K, value: V) -> Option<V> {
        if self.capacity == 0 {
            return Some(value);
        }

        // Update or insert
        if self.map.contains_key(&key) {
            // Update existing entry
            let old_value = self.map.insert(key.clone(), value);
            let freq = self.frequencies.entry(key.clone()).or_insert(0);
            *freq += 1;
            self.freq_heap.push((*freq, key));
            old_value
        } else {
            // Need to evict if at capacity
            while self.map.len() >= self.capacity {
                self.evict_mfu();
            }

            // Insert new entry
            self.map.insert(key.clone(), value);
            self.frequencies.insert(key.clone(), 1);
            self.freq_heap.push((1, key));
            None
        }
    }

    fn contains(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    fn len(&self) -> usize {
        self.map.len()
    }

    fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn clear(&mut self) {
        self.map.clear();
        self.frequencies.clear();
        self.freq_heap.clear();
    }
}

impl<K, V> fmt::Debug for MfuCore<K, V>
where
    K: fmt::Debug + Eq + Hash,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MfuCore")
            .field("len", &self.map.len())
            .field("capacity", &self.capacity)
            .field("frequencies", &self.frequencies)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_cache_is_empty() {
        let cache: MfuCore<i32, i32> = MfuCore::new(10);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.capacity(), 10);
    }

    #[test]
    fn insert_and_get() {
        let mut cache = MfuCore::new(3);
        cache.insert(1, 100);
        cache.insert(2, 200);
        assert_eq!(cache.get(&1), Some(&100));
        assert_eq!(cache.get(&2), Some(&200));
        assert_eq!(cache.get(&3), None);
    }

    #[test]
    fn mfu_eviction() {
        let mut cache = MfuCore::new(3);

        // Insert 3 items
        cache.insert(1, 100);
        cache.insert(2, 200);
        cache.insert(3, 300);

        // Access item 1 many times (highest frequency)
        for _ in 0..10 {
            cache.get(&1);
        }

        // Access item 2 a few times
        cache.get(&2);
        cache.get(&2);

        // Item 3 has frequency 1, item 2 has frequency 3, item 1 has frequency 11
        assert_eq!(cache.frequency(&1), Some(11));
        assert_eq!(cache.frequency(&2), Some(3));
        assert_eq!(cache.frequency(&3), Some(1));

        // Insert new item, should evict item 1 (highest frequency)
        cache.insert(4, 400);

        assert!(!cache.contains(&1)); // Evicted (most frequently used)
        assert!(cache.contains(&2));
        assert!(cache.contains(&3));
        assert!(cache.contains(&4));
    }

    #[test]
    fn frequency_tracking() {
        let mut cache = MfuCore::new(5);
        cache.insert(1, 100);

        assert_eq!(cache.frequency(&1), Some(1));

        cache.get(&1);
        assert_eq!(cache.frequency(&1), Some(2));

        cache.get(&1);
        cache.get(&1);
        assert_eq!(cache.frequency(&1), Some(4));
    }

    #[test]
    fn pop_mfu() {
        let mut cache = MfuCore::new(3);
        cache.insert(1, 100);
        cache.insert(2, 200);
        cache.insert(3, 300);

        // Access to create different frequencies
        cache.get(&1); // freq 2
        cache.get(&1); // freq 3
        cache.get(&2); // freq 2

        assert_eq!(cache.frequency(&1), Some(3));
        assert_eq!(cache.frequency(&2), Some(2));
        assert_eq!(cache.frequency(&3), Some(1));

        // Pop MFU (item 1 with freq 3)
        let (key, value) = cache.pop_mfu().unwrap();
        assert_eq!(key, 1);
        assert_eq!(value, 100);
        assert!(!cache.contains(&1));
    }

    #[test]
    fn peek_mfu() {
        let mut cache = MfuCore::new(3);
        cache.insert(1, 100);
        cache.insert(2, 200);
        cache.insert(3, 300);

        cache.get(&2);
        cache.get(&2);
        cache.get(&2); // freq 4

        let (key, value) = cache.peek_mfu().unwrap();
        assert_eq!(*key, 2);
        assert_eq!(*value, 200);

        // Peek doesn't remove
        assert!(cache.contains(&2));
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn update_existing_key() {
        let mut cache = MfuCore::new(3);
        cache.insert(1, 100);
        let old_value = cache.insert(1, 999); // Update

        assert_eq!(old_value, Some(100));
        assert_eq!(cache.get(&1), Some(&999));
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.frequency(&1), Some(3)); // 2 inserts + 1 get = 3
    }

    #[test]
    fn clear() {
        let mut cache = MfuCore::new(3);
        cache.insert(1, 100);
        cache.insert(2, 200);
        cache.insert(3, 300);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert!(!cache.contains(&1));
        assert_eq!(cache.frequency(&1), None);
    }

    #[test]
    fn zero_capacity_cache() {
        let mut cache = MfuCore::new(0);
        cache.insert(1, 100);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn heap_rebuild_on_stale_accumulation() {
        let mut cache = MfuCore::new(2);
        cache.insert(1, 100);
        cache.insert(2, 200);

        // Access item 1 many times to accumulate stale entries
        for _ in 0..20 {
            cache.get(&1);
        }

        // Heap should eventually rebuild
        assert_eq!(cache.len(), 2);
        assert!(cache.contains(&1));
        assert!(cache.contains(&2));
    }

    #[test]
    fn evict_on_equal_frequencies() {
        let mut cache = MfuCore::new(2);
        cache.insert(1, 100);
        cache.insert(2, 200);

        // Both have freq 1
        assert_eq!(cache.frequency(&1), Some(1));
        assert_eq!(cache.frequency(&2), Some(1));

        // Insert should evict one of them (heap order dependent)
        cache.insert(3, 300);
        assert_eq!(cache.len(), 2);
        assert!(cache.contains(&3));
    }

    #[test]
    fn core_cache_trait() {
        let mut cache = MfuCore::new(3);
        cache.insert(1, 100);
        assert_eq!(cache.get(&1), Some(&100));
        assert!(cache.contains(&1));
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.capacity(), 3);
    }

    #[test]
    fn mfu_vs_lfu_behavior() {
        // Demonstrate difference between MFU and LFU
        let mut cache = MfuCore::new(3);

        cache.insert(1, 100);
        cache.insert(2, 200);
        cache.insert(3, 300);

        // Make item 1 very frequent
        for _ in 0..100 {
            cache.get(&1);
        }

        // Make item 2 moderately frequent
        for _ in 0..10 {
            cache.get(&2);
        }

        // Item 3 accessed least (freq 1)
        // MFU will evict item 1 (freq 101), opposite of LFU
        cache.insert(4, 400);

        assert!(!cache.contains(&1)); // Item 1 evicted (most frequent!)
        assert!(cache.contains(&2));
        assert!(cache.contains(&3)); // Item 3 kept (least frequent)
        assert!(cache.contains(&4));
    }

    #[test]
    fn burst_workload() {
        // Simulate burst where MFU might be useful
        let mut cache = MfuCore::new(3);

        // Initial state
        cache.insert(1, 100);
        cache.insert(2, 200);
        cache.insert(3, 300);

        // Burst on item 1 (temporary high frequency)
        for _ in 0..50 {
            cache.get(&1);
        }

        // Now item 1 is "hot" from burst, MFU will evict it
        cache.insert(4, 400);

        // MFU evicted the burst item, keeping lower frequency items
        assert!(!cache.contains(&1)); // Burst item evicted
        assert!(cache.contains(&2));
        assert!(cache.contains(&3));
        assert!(cache.contains(&4));
    }

    #[test]
    fn string_keys() {
        let mut cache = MfuCore::new(3);
        cache.insert("a".to_string(), "alpha");
        cache.insert("b".to_string(), "beta");
        cache.insert("c".to_string(), "gamma");

        cache.get(&"a".to_string());
        cache.get(&"a".to_string());
        cache.get(&"a".to_string());

        assert_eq!(cache.frequency(&"a".to_string()), Some(4));

        cache.insert("d".to_string(), "delta");
        assert!(!cache.contains(&"a".to_string())); // Most frequent, evicted
    }

    #[test]
    fn debug_format() {
        let mut cache = MfuCore::new(3);
        cache.insert(1, 100);
        cache.insert(2, 200);

        let debug_str = format!("{:?}", cache);
        assert!(debug_str.contains("MfuCore"));
        assert!(debug_str.contains("len"));
        assert!(debug_str.contains("capacity"));
    }

    #[test]
    #[cfg(debug_assertions)]
    fn validate_invariants_after_operations() {
        let mut cache = MfuCore::new(5);

        // Insert items
        for i in 1..=5 {
            cache.insert(i, i * 100);
        }
        cache.validate_invariants();

        // Access items to create different frequencies
        for _ in 0..10 {
            cache.get(&1);
        }
        for _ in 0..5 {
            cache.get(&2);
        }
        cache.validate_invariants();

        // Trigger evictions
        cache.insert(6, 600);
        cache.validate_invariants();

        cache.insert(7, 700);
        cache.validate_invariants();

        // Pop MFU
        cache.pop_mfu();
        cache.validate_invariants();

        // Clear
        cache.clear();
        cache.validate_invariants();

        // Verify empty state
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.frequencies.len(), 0);
    }

    #[test]
    #[cfg(debug_assertions)]
    fn validate_invariants_with_heap_rebuild() {
        let mut cache = MfuCore::new(3);
        cache.insert(1, 100);
        cache.insert(2, 200);
        cache.insert(3, 300);

        // Access many times to accumulate stale heap entries
        for _ in 0..50 {
            cache.get(&1);
        }
        cache.validate_invariants();

        // Should trigger heap rebuild
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.frequencies.len(), 3);
    }
}
