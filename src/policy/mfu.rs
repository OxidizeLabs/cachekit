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
//!   │   │  freq_heap: BinaryHeap<HeapEntry<K>>  (Max-Heap by frequency)      │ │
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
//! - For concurrent access, wrap in external synchronization (e.g., `Arc<RwLock<MfuCore>>`)
//!
//! ## Implementation Notes
//!
//! - Uses max-heap via `HeapEntry` (orders by frequency, then sequence number)
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

#[cfg(feature = "metrics")]
use crate::metrics::metrics_impl::MfuMetrics;
#[cfg(feature = "metrics")]
use crate::metrics::snapshot::MfuMetricsSnapshot;
#[cfg(feature = "metrics")]
use crate::metrics::traits::{
    CoreMetricsRecorder, MetricsSnapshotProvider, MfuMetricsReadRecorder, MfuMetricsRecorder,
};
use crate::traits::{Cache, EvictingCache, FrequencyTracking, VictimInspectable};
use rustc_hash::FxHashMap;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fmt;
use std::hash::Hash;

/// Max-heap rebuilds when stale entries exceed this factor times live entries.
const HEAP_REBUILD_FACTOR: usize = 3;

/// Heap entry that orders by frequency (descending), breaking ties by insertion
/// sequence. Avoids requiring `K: Ord`.
#[derive(Clone)]
struct HeapEntry<K> {
    freq: u64,
    seq: u64,
    key: K,
}

impl<K> PartialEq for HeapEntry<K> {
    fn eq(&self, other: &Self) -> bool {
        self.freq == other.freq && self.seq == other.seq
    }
}

impl<K> Eq for HeapEntry<K> {}

impl<K> PartialOrd for HeapEntry<K> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<K> Ord for HeapEntry<K> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.freq
            .cmp(&other.freq)
            .then_with(|| self.seq.cmp(&other.seq))
    }
}

/// MFU cache core that evicts the most frequently used entry.
///
/// Implements [`Cache`] for generic cache access, plus
/// [`EvictingCache`], [`VictimInspectable`], and [`FrequencyTracking`]
/// capability traits. Uses a [`BinaryHeap`] with `HeapEntry` wrappers
/// for O(log n) eviction of the highest-frequency entry.
pub struct MfuCore<K, V> {
    map: FxHashMap<K, V>,
    frequencies: FxHashMap<K, u64>,
    freq_heap: BinaryHeap<HeapEntry<K>>,
    capacity: usize,
    seq: u64,
    #[cfg(feature = "metrics")]
    metrics: MfuMetrics,
}

impl<K, V> MfuCore<K, V>
where
    K: Clone + Eq + Hash,
{
    /// Creates a new MFU cache with the specified capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            map: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
            frequencies: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
            freq_heap: BinaryHeap::with_capacity(capacity),
            capacity,
            seq: 0,
            #[cfg(feature = "metrics")]
            metrics: MfuMetrics::default(),
        }
    }

    /// Gets a value by key, incrementing its frequency.
    pub fn get(&mut self, key: &K) -> Option<&V> {
        if self.map.contains_key(key) {
            #[cfg(feature = "metrics")]
            self.metrics.record_get_hit();

            let freq = self.frequencies.entry(key.clone()).or_insert(0);
            *freq += 1;

            self.seq += 1;
            self.freq_heap.push(HeapEntry {
                freq: *freq,
                seq: self.seq,
                key: key.clone(),
            });

            if self.freq_heap.len() > self.map.len() * HEAP_REBUILD_FACTOR {
                self.rebuild_heap();
            }

            self.map.get(key)
        } else {
            #[cfg(feature = "metrics")]
            self.metrics.record_get_miss();
            None
        }
    }

    /// Inserts a key-value pair, evicting the most frequently used entry if at capacity.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        #[cfg(feature = "metrics")]
        self.metrics.record_insert_call();

        if self.capacity == 0 {
            return None;
        }

        let result = if self.map.contains_key(&key) {
            #[cfg(feature = "metrics")]
            self.metrics.record_insert_update();

            let old_value = self.map.insert(key.clone(), value);
            let freq = self.frequencies.entry(key.clone()).or_insert(0);
            *freq += 1;
            self.seq += 1;
            self.freq_heap.push(HeapEntry {
                freq: *freq,
                seq: self.seq,
                key,
            });
            old_value
        } else {
            #[cfg(feature = "metrics")]
            self.metrics.record_insert_new();

            #[cfg(feature = "metrics")]
            if self.map.len() >= self.capacity {
                self.metrics.record_evict_call();
            }

            while self.map.len() >= self.capacity {
                self.evict_mfu();
                #[cfg(feature = "metrics")]
                self.metrics.record_evicted_entry();
            }

            self.map.insert(key.clone(), value);
            self.frequencies.insert(key.clone(), 1);
            self.seq += 1;
            self.freq_heap.push(HeapEntry {
                freq: 1,
                seq: self.seq,
                key,
            });
            None
        };

        #[cfg(debug_assertions)]
        self.validate_invariants();

        result
    }

    /// Evicts the entry with the highest frequency (MFU).
    fn evict_mfu(&mut self) {
        while let Some(entry) = self.freq_heap.pop() {
            if let Some(&current_freq) = self.frequencies.get(&entry.key) {
                if current_freq == entry.freq {
                    self.map.remove(&entry.key);
                    self.frequencies.remove(&entry.key);
                    if self.map.is_empty() {
                        self.freq_heap.clear();
                    }

                    #[cfg(debug_assertions)]
                    self.validate_invariants();

                    return;
                }
            }
        }

        if !self.map.is_empty() {
            self.rebuild_heap();
            if let Some(entry) = self.freq_heap.pop() {
                self.map.remove(&entry.key);
                self.frequencies.remove(&entry.key);
                if self.map.is_empty() {
                    self.freq_heap.clear();
                }

                #[cfg(debug_assertions)]
                self.validate_invariants();
            }
        }
    }

    /// Rebuilds the heap from current frequencies, dropping all stale entries.
    fn rebuild_heap(&mut self) {
        self.freq_heap.clear();
        for (key, &freq) in &self.frequencies {
            self.seq += 1;
            self.freq_heap.push(HeapEntry {
                freq,
                seq: self.seq,
                key: key.clone(),
            });
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

    /// Returns an iterator over all key-value pairs.
    pub fn iter(&self) -> std::collections::hash_map::Iter<'_, K, V> {
        self.map.iter()
    }

    /// Returns an iterator over all keys.
    pub fn keys(&self) -> std::collections::hash_map::Keys<'_, K, V> {
        self.map.keys()
    }

    /// Returns an iterator over all values.
    pub fn values(&self) -> std::collections::hash_map::Values<'_, K, V> {
        self.map.values()
    }

    /// Checks if a key exists in the cache without updating frequency.
    pub fn contains(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    /// Peeks at a value without updating frequency.
    pub fn peek(&self, key: &K) -> Option<&V> {
        self.map.get(key)
    }

    /// Removes all entries from the cache.
    pub fn clear(&mut self) {
        #[cfg(feature = "metrics")]
        self.metrics.record_clear();

        self.map.clear();
        self.frequencies.clear();
        self.freq_heap.clear();

        #[cfg(debug_assertions)]
        self.validate_invariants();
    }

    /// Removes a key from the cache, returning its value if present.
    ///
    /// Stale heap entries for the removed key are cleaned up lazily
    /// during subsequent evictions or heap rebuilds.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        self.frequencies.remove(key);
        let result = self.map.remove(key);
        if result.is_some() && self.map.is_empty() {
            self.freq_heap.clear();
        }

        #[cfg(debug_assertions)]
        if result.is_some() {
            self.validate_invariants();
        }

        result
    }

    /// Gets the current frequency count for a key.
    pub fn frequency(&self, key: &K) -> Option<u64> {
        #[cfg(feature = "metrics")]
        (&self.metrics).record_frequency_call();

        let result = self.frequencies.get(key).copied();

        #[cfg(feature = "metrics")]
        if result.is_some() {
            (&self.metrics).record_frequency_found();
        }

        result
    }

    /// Removes and returns the entry with the highest frequency.
    #[must_use]
    pub fn pop_mfu(&mut self) -> Option<(K, V)> {
        #[cfg(feature = "metrics")]
        self.metrics.record_pop_mfu_call();

        while let Some(entry) = self.freq_heap.pop() {
            if let Some(&current_freq) = self.frequencies.get(&entry.key) {
                if current_freq == entry.freq {
                    if let Some(value) = self.map.remove(&entry.key) {
                        self.frequencies.remove(&entry.key);
                        if self.map.is_empty() {
                            self.freq_heap.clear();
                        }

                        #[cfg(feature = "metrics")]
                        self.metrics.record_pop_mfu_found();

                        #[cfg(debug_assertions)]
                        self.validate_invariants();

                        return Some((entry.key, value));
                    }
                }
            }
        }
        None
    }

    /// Peeks at the entry with highest frequency without removing it.
    #[must_use]
    pub fn peek_mfu(&self) -> Option<(&K, &V)> {
        #[cfg(feature = "metrics")]
        (&self.metrics).record_peek_mfu_call();

        let mut max_freq = 0u64;
        let mut max_key: Option<&K> = None;

        for (key, &freq) in &self.frequencies {
            if freq > max_freq {
                max_freq = freq;
                max_key = Some(key);
            }
        }

        let result = max_key.and_then(|k| self.map.get(k).map(|v| (k, v)));

        #[cfg(feature = "metrics")]
        if result.is_some() {
            (&self.metrics).record_peek_mfu_found();
        }

        result
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
        debug_assert_eq!(
            self.map.len(),
            self.frequencies.len(),
            "Map and frequencies have different sizes: map={}, freq={}",
            self.map.len(),
            self.frequencies.len()
        );

        for key in self.map.keys() {
            debug_assert!(
                self.frequencies.contains_key(key),
                "Key in map but not in frequencies"
            );
        }

        for key in self.frequencies.keys() {
            debug_assert!(
                self.map.contains_key(key),
                "Key in frequencies but not in map"
            );
        }

        for &freq in self.frequencies.values() {
            debug_assert!(freq >= 1, "Invalid frequency found: {}", freq);
        }

        debug_assert!(
            self.freq_heap.len() <= self.map.len() * (HEAP_REBUILD_FACTOR + 1),
            "Heap size {} exceeds reasonable bounds for map size {}",
            self.freq_heap.len(),
            self.map.len()
        );
    }
}

impl<K, V> Cache<K, V> for MfuCore<K, V>
where
    K: Clone + Eq + Hash,
{
    fn contains(&self, key: &K) -> bool {
        MfuCore::contains(self, key)
    }

    fn len(&self) -> usize {
        MfuCore::len(self)
    }

    fn capacity(&self) -> usize {
        MfuCore::capacity(self)
    }

    fn peek(&self, key: &K) -> Option<&V> {
        MfuCore::peek(self, key)
    }

    fn get(&mut self, key: &K) -> Option<&V> {
        MfuCore::get(self, key)
    }

    fn insert(&mut self, key: K, value: V) -> Option<V> {
        MfuCore::insert(self, key, value)
    }

    fn remove(&mut self, key: &K) -> Option<V> {
        MfuCore::remove(self, key)
    }

    fn clear(&mut self) {
        MfuCore::clear(self)
    }
}

impl<K, V> EvictingCache<K, V> for MfuCore<K, V>
where
    K: Clone + Eq + Hash,
{
    fn evict_one(&mut self) -> Option<(K, V)> {
        self.pop_mfu()
    }
}

impl<K, V> VictimInspectable<K, V> for MfuCore<K, V>
where
    K: Clone + Eq + Hash,
{
    fn peek_victim(&self) -> Option<(&K, &V)> {
        self.peek_mfu()
    }
}

impl<K, V> FrequencyTracking<K, V> for MfuCore<K, V>
where
    K: Clone + Eq + Hash,
{
    fn frequency(&self, key: &K) -> Option<u64> {
        MfuCore::frequency(self, key)
    }
}

impl<K, V> fmt::Debug for MfuCore<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MfuCore")
            .field("len", &self.map.len())
            .field("capacity", &self.capacity)
            .field("heap_entries", &self.freq_heap.len())
            .finish()
    }
}

impl<K, V> Clone for MfuCore<K, V>
where
    K: Clone + Eq + Hash,
    V: Clone,
{
    fn clone(&self) -> Self {
        Self {
            map: self.map.clone(),
            frequencies: self.frequencies.clone(),
            freq_heap: self.freq_heap.clone(),
            capacity: self.capacity,
            seq: self.seq,
            #[cfg(feature = "metrics")]
            metrics: MfuMetrics::default(),
        }
    }
}

impl<K, V> Default for MfuCore<K, V>
where
    K: Clone + Eq + Hash,
{
    /// Returns a zero-capacity cache that rejects all insertions.
    fn default() -> Self {
        Self::new(0)
    }
}

impl<K, V> IntoIterator for MfuCore<K, V>
where
    K: Clone + Eq + Hash,
{
    type Item = (K, V);
    type IntoIter = std::collections::hash_map::IntoIter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.map.into_iter()
    }
}

impl<'a, K, V> IntoIterator for &'a MfuCore<K, V>
where
    K: Clone + Eq + Hash,
{
    type Item = (&'a K, &'a V);
    type IntoIter = std::collections::hash_map::Iter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.map.iter()
    }
}

#[cfg(feature = "metrics")]
impl<K, V> MfuCore<K, V>
where
    K: Clone + Eq + Hash,
{
    /// Returns a snapshot of cache metrics.
    pub fn metrics_snapshot(&self) -> MfuMetricsSnapshot {
        MfuMetricsSnapshot {
            get_calls: self.metrics.get_calls,
            get_hits: self.metrics.get_hits,
            get_misses: self.metrics.get_misses,
            insert_calls: self.metrics.insert_calls,
            insert_updates: self.metrics.insert_updates,
            insert_new: self.metrics.insert_new,
            evict_calls: self.metrics.evict_calls,
            evicted_entries: self.metrics.evicted_entries,
            pop_mfu_calls: self.metrics.pop_mfu_calls,
            pop_mfu_found: self.metrics.pop_mfu_found,
            peek_mfu_calls: self.metrics.peek_mfu_calls.get(),
            peek_mfu_found: self.metrics.peek_mfu_found.get(),
            frequency_calls: self.metrics.frequency_calls.get(),
            frequency_found: self.metrics.frequency_found.get(),
            cache_len: self.len(),
            capacity: self.capacity,
        }
    }
}

#[cfg(feature = "metrics")]
impl<K, V> MetricsSnapshotProvider<MfuMetricsSnapshot> for MfuCore<K, V>
where
    K: Clone + Eq + Hash,
{
    fn snapshot(&self) -> MfuMetricsSnapshot {
        self.metrics_snapshot()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const _: () = {
        fn _assert_send_sync<T: Send + Sync>() {}
        fn _check() {
            _assert_send_sync::<MfuCore<String, String>>();
        }
    };

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

        cache.insert(1, 100);
        cache.insert(2, 200);
        cache.insert(3, 300);

        for _ in 0..10 {
            cache.get(&1);
        }

        cache.get(&2);
        cache.get(&2);

        assert_eq!(cache.frequency(&1), Some(11));
        assert_eq!(cache.frequency(&2), Some(3));
        assert_eq!(cache.frequency(&3), Some(1));

        cache.insert(4, 400);

        assert!(!cache.contains(&1));
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

        cache.get(&1); // freq 2
        cache.get(&1); // freq 3
        cache.get(&2); // freq 2

        assert_eq!(cache.frequency(&1), Some(3));
        assert_eq!(cache.frequency(&2), Some(2));
        assert_eq!(cache.frequency(&3), Some(1));

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

        assert!(cache.contains(&2));
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn update_existing_key() {
        let mut cache = MfuCore::new(3);
        cache.insert(1, 100);
        let old_value = cache.insert(1, 999);

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

        for _ in 0..20 {
            cache.get(&1);
        }

        assert_eq!(cache.len(), 2);
        assert!(cache.contains(&1));
        assert!(cache.contains(&2));
    }

    #[test]
    fn evict_on_equal_frequencies() {
        let mut cache = MfuCore::new(2);
        cache.insert(1, 100);
        cache.insert(2, 200);

        assert_eq!(cache.frequency(&1), Some(1));
        assert_eq!(cache.frequency(&2), Some(1));

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
        let mut cache = MfuCore::new(3);

        cache.insert(1, 100);
        cache.insert(2, 200);
        cache.insert(3, 300);

        for _ in 0..100 {
            cache.get(&1);
        }

        for _ in 0..10 {
            cache.get(&2);
        }

        cache.insert(4, 400);

        assert!(!cache.contains(&1));
        assert!(cache.contains(&2));
        assert!(cache.contains(&3));
        assert!(cache.contains(&4));
    }

    #[test]
    fn burst_workload() {
        let mut cache = MfuCore::new(3);

        cache.insert(1, 100);
        cache.insert(2, 200);
        cache.insert(3, 300);

        for _ in 0..50 {
            cache.get(&1);
        }

        cache.insert(4, 400);

        assert!(!cache.contains(&1));
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
        assert!(!cache.contains(&"a".to_string()));
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
        assert!(debug_str.contains("heap_entries"));
    }

    #[test]
    #[cfg(debug_assertions)]
    fn validate_invariants_after_operations() {
        let mut cache = MfuCore::new(5);

        for i in 1..=5 {
            cache.insert(i, i * 100);
        }
        cache.validate_invariants();

        for _ in 0..10 {
            cache.get(&1);
        }
        for _ in 0..5 {
            cache.get(&2);
        }
        cache.validate_invariants();

        cache.insert(6, 600);
        cache.validate_invariants();

        cache.insert(7, 700);
        cache.validate_invariants();

        let _ = cache.pop_mfu();
        cache.validate_invariants();

        cache.clear();
        cache.validate_invariants();

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

        for _ in 0..50 {
            cache.get(&1);
        }
        cache.validate_invariants();

        assert_eq!(cache.len(), 3);
        assert_eq!(cache.frequencies.len(), 3);
    }

    // ==============================================
    // Regression Tests
    // ==============================================

    #[test]
    fn zero_capacity_insert_returns_none() {
        let mut cache: MfuCore<&str, i32> = MfuCore::new(0);

        let result = cache.insert("new_key", 42);

        assert_eq!(
            result, None,
            "MfuCore::insert at capacity=0 should return None for a new key"
        );
    }

    // ==============================================
    // New: remove, Clone, Default, IntoIterator
    // ==============================================

    #[test]
    fn remove_existing_key() {
        let mut cache = MfuCore::new(5);
        cache.insert(1, 100);
        cache.insert(2, 200);
        cache.insert(3, 300);

        assert_eq!(cache.remove(&2), Some(200));
        assert_eq!(cache.len(), 2);
        assert!(!cache.contains(&2));
        assert_eq!(cache.frequency(&2), None);
    }

    #[test]
    fn remove_nonexistent_key() {
        let mut cache = MfuCore::new(5);
        cache.insert(1, 100);

        assert_eq!(cache.remove(&99), None);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn remove_then_insert() {
        let mut cache = MfuCore::new(3);
        cache.insert(1, 100);
        cache.insert(2, 200);
        cache.insert(3, 300);

        cache.remove(&2);
        assert_eq!(cache.len(), 2);

        cache.insert(4, 400);
        assert_eq!(cache.len(), 3);
        assert!(cache.contains(&1));
        assert!(cache.contains(&3));
        assert!(cache.contains(&4));
    }

    #[test]
    fn cache_trait_remove() {
        let mut cache = MfuCore::new(5);
        cache.insert(1, 100);
        cache.insert(2, 200);

        assert_eq!(Cache::remove(&mut cache, &1), Some(100));
        assert!(!cache.contains(&1));
    }

    #[test]
    fn clone_cache() {
        let mut cache = MfuCore::new(3);
        cache.insert(1, 100);
        cache.insert(2, 200);
        cache.get(&1);

        let cloned = cache.clone();
        assert_eq!(cloned.len(), 2);
        assert_eq!(cloned.capacity(), 3);
        assert!(cloned.contains(&1));
        assert!(cloned.contains(&2));
        assert_eq!(cloned.frequency(&1), Some(2));
    }

    #[test]
    fn clone_independence() {
        let mut cache = MfuCore::new(3);
        cache.insert(1, 100);

        let mut cloned = cache.clone();
        cloned.insert(2, 200);

        assert_eq!(cache.len(), 1);
        assert_eq!(cloned.len(), 2);
    }

    #[test]
    fn default_cache() {
        let cache: MfuCore<i32, i32> = MfuCore::default();
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.capacity(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn into_iterator_owned() {
        let mut cache = MfuCore::new(3);
        cache.insert(1, 100);
        cache.insert(2, 200);

        let mut entries: Vec<_> = cache.into_iter().collect();
        entries.sort_by_key(|(k, _)| *k);
        assert_eq!(entries, vec![(1, 100), (2, 200)]);
    }

    #[test]
    fn into_iterator_borrowed() {
        let mut cache = MfuCore::new(3);
        cache.insert(1, 100);
        cache.insert(2, 200);

        let mut entries: Vec<_> = (&cache).into_iter().map(|(&k, &v)| (k, v)).collect();
        entries.sort_by_key(|(k, _)| *k);
        assert_eq!(entries, vec![(1, 100), (2, 200)]);
    }

    #[test]
    fn iter_keys_values() {
        let mut cache = MfuCore::new(3);
        cache.insert(1, 100);
        cache.insert(2, 200);

        assert_eq!(cache.iter().count(), 2);
        assert_eq!(cache.keys().count(), 2);
        assert_eq!(cache.values().count(), 2);
    }

    #[test]
    fn non_ord_key_compiles() {
        #[derive(Clone, Eq, PartialEq, Hash, Debug)]
        struct MyKey(Vec<u8>);

        let mut cache: MfuCore<MyKey, i32> = MfuCore::new(3);
        cache.insert(MyKey(vec![1, 2]), 100);
        cache.insert(MyKey(vec![3, 4]), 200);
        assert_eq!(cache.get(&MyKey(vec![1, 2])), Some(&100));
        assert_eq!(cache.len(), 2);
    }
}
