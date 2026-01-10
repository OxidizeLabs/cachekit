//! # LRU-K Cache Implementation
//!
//! This module provides an implementation of the LRU-K replacement policy (specifically LRU-2
//! by default). LRU-K improves upon standard LRU by tracking the K-th most recent access time,
//! providing resistance to cache pollution from sequential scans.
//!
//! ## Architecture
//!
//! ```text
//!   ┌──────────────────────────────────────────────────────────────────────────┐
//!   │                          LRUKCache<K, V>                                 │
//!   │                                                                          │
//!   │   ┌────────────────────────────────────────────────────────────────────┐ │
//!   │   │  HashMap<K, (V, VecDeque<u64>)>                                    │ │
//!   │   │                                                                    │ │
//!   │   │  ┌─────────┬───────────────────────────────────────────────────┐   │ │
//!   │   │  │   Key   │  (Value, Access History)                          │   │ │
//!   │   │  ├─────────┼───────────────────────────────────────────────────┤   │ │
//!   │   │  │ page_1  │  (data, [t₁, t₅, t₉])  ← 3 accesses, K=3          │   │ │
//!   │   │  │ page_2  │  (data, [t₃])          ← 1 access (< K)           │   │ │
//!   │   │  │ page_3  │  (data, [t₂, t₇])      ← 2 accesses (< K if K=3)  │   │ │
//!   │   │  └─────────┴───────────────────────────────────────────────────┘   │ │
//!   │   │                                                                    │ │
//!   │   │  VecDeque stores last K timestamps (microseconds since epoch)      │ │
//!   │   └────────────────────────────────────────────────────────────────────┘ │
//!   │                                                                          │
//!   │   Configuration:                                                         │
//!   │   • capacity: Maximum entries                                            │
//!   │   • k: Number of accesses to track (default: 2)                          │
//!   └──────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## LRU-K Eviction Policy
//!
//! ```text
//!   Eviction Priority (highest to lowest):
//!   ═══════════════════════════════════════════════════════════════════════════
//!
//!   PRIORITY 1: Items with fewer than K accesses (< K)
//!   ─────────────────────────────────────────────────────────────────────────────
//!     • These items haven't proven their "hotness"
//!     • Among them, evict the one with the EARLIEST first access
//!
//!     Example (K=2):
//!       page_A: [t₁]        ← 1 access, earliest = t₁  ← EVICT THIS
//!       page_B: [t₃]        ← 1 access, earliest = t₃
//!
//!   PRIORITY 2: Items with K or more accesses (≥ K)
//!   ─────────────────────────────────────────────────────────────────────────────
//!     • Only considered if ALL items have ≥ K accesses
//!     • Evict the one with the OLDEST K-th most recent access (backward K-distance)
//!
//!     Example (K=2):
//!       page_C: [t₂, t₈]    ← K-distance = t₂  ← EVICT THIS (oldest K-dist)
//!       page_D: [t₅, t₉]    ← K-distance = t₅
//!
//!   ═══════════════════════════════════════════════════════════════════════════
//!
//!   K-Distance Calculation:
//!
//!     History: [t_oldest, ..., t_recent]   (VecDeque, front=oldest)
//!     K-distance = history[len - K]        (K-th from the end)
//!
//!     Example (K=2, history=[t₁, t₅, t₉]):
//!       len = 3
//!       K-distance index = 3 - 2 = 1
//!       K-distance = t₅
//! ```
//!
//! ## Scan Resistance Explained
//!
//! ```text
//!   Problem with standard LRU:
//!   ═══════════════════════════════════════════════════════════════════════════
//!
//!   Cache: [A, B, C, D]  (A = MRU, D = LRU)
//!
//!   Sequential scan reads pages X₁, X₂, X₃, X₄ (one-time access each):
//!
//!     After X₁:  [X₁, A, B, C]  ← D evicted
//!     After X₂:  [X₂, X₁, A, B] ← C evicted
//!     After X₃:  [X₃, X₂, X₁, A] ← B evicted
//!     After X₄:  [X₄, X₃, X₂, X₁] ← A evicted  ← ALL hot pages gone!
//!
//!   ═══════════════════════════════════════════════════════════════════════════
//!
//!   LRU-K (K=2) solution:
//!   ═══════════════════════════════════════════════════════════════════════════
//!
//!   Cache (with access counts):
//!     A: 5 accesses (K-dist = t₁₀)  ← "hot" page
//!     B: 3 accesses (K-dist = t₈)   ← "hot" page
//!     C: 2 accesses (K-dist = t₅)   ← "warm" page
//!     D: 1 access   (< K)           ← "cold" page
//!
//!   Sequential scan reads X₁:
//!     X₁ has 1 access (< K)
//!     D also has 1 access (< K)
//!     X₁ is newer than D → D is evicted (not the hot pages!)
//!
//!   Result: Hot pages A, B, C survive the scan!
//! ```
//!
//! ## Key Components
//!
//! | Component        | Description                                        |
//! |------------------|----------------------------------------------------|
//! | `LRUKCache<K,V>` | Main cache struct with capacity and K value        |
//! | `index`          | `HashMap<K, usize>` to slot indices                |
//! | `cold`/`hot`      | Segmented LRU lists (<K and >=K accesses)          |
//! | `capacity`       | Maximum number of entries                          |
//! | `k`              | Number of accesses to track (default: 2)           |
//!
//! ## Core Operations (CoreCache + MutableCache + LRUKCacheTrait)
//!
//! | Method              | Complexity | Description                              |
//! |---------------------|------------|------------------------------------------|
//! | `new(capacity)`     | O(1)       | Create cache with K=2 (default)          |
//! | `with_k(cap, k)`    | O(1)       | Create cache with custom K value         |
//! | `insert(key, val)`  | O(1)*      | Insert/update, may trigger O(1) eviction |
//! | `get(&key)`         | O(1)       | Get value, updates access history        |
//! | `contains(&key)`    | O(1)       | Check if key exists                      |
//! | `remove(&key)`      | O(1)       | Remove entry by key                      |
//! | `len()`             | O(1)       | Current number of entries                |
//! | `capacity()`        | O(1)       | Maximum capacity                         |
//! | `clear()`           | O(N)       | Remove all entries                       |
//!
//! ## LRU-K Specific Operations (LRUKCacheTrait)
//!
//! | Method               | Complexity | Description                             |
//! |----------------------|------------|-----------------------------------------|
//! | `pop_lru_k()`        | O(1)       | Remove and return victim entry          |
//! | `peek_lru_k()`       | O(1)       | Peek at victim without removing         |
//! | `k_value()`          | O(1)       | Get the K value                         |
//! | `access_history()`   | O(K)       | Get timestamps (most recent first)      |
//! | `access_count()`     | O(1)       | Get number of accesses for key          |
//! | `k_distance()`       | O(1)       | Get K-distance (None if < K accesses)   |
//! | `touch(&key)`        | O(1)       | Update access time without getting      |
//! | `k_distance_rank()`  | O(N log N) | Get eviction priority rank              |
//!
//! ## Performance Characteristics
//!
//! | Operation              | Time       | Notes                              |
//! |------------------------|------------|------------------------------------|
//! | `get`, `insert` (hit)  | O(1)       | HashMap lookup + VecDeque update   |
//! | `insert` (eviction)    | O(1)       | Bucketed by cold/hot lists         |
//! | `pop_lru_k`            | O(1)       | Tail lookup on cold/hot list       |
//! | `peek_lru_k`           | O(1)       | Tail lookup on cold/hot list       |
//! | `k_distance_rank`      | O(N log N) | Collects and sorts all entries     |
//! | Per-entry overhead     | ~24 bytes  | VecDeque + K × 8 bytes timestamps  |
//!
//! ## Design Rationale
//!
//! - **Scan Resistance**: Standard LRU flushes entire cache on sequential scans.
//!   LRU-K requires K accesses before an item is considered "hot".
//! - **Segmented Queues**: Cold entries (<K) are FIFO; hot entries (>=K) are LRU.
//! - **Predictability**: O(1) eviction paths with bounded list operations.
//!
//! ## Trade-offs
//!
//! | Aspect           | Pros                               | Cons                            |
//! |------------------|------------------------------------|---------------------------------|
//! | Hit Ratio        | Better than LRU for DB workloads   | Overhead for simple patterns    |
//! | Scan Resistance  | Excellent (core feature)           | -                               |
//! | Eviction Time    | -                                  | O(1) list operations            |
//! | Memory           | Bounded history (K timestamps)     | Extra ~24 + 8K bytes per entry  |
//! | Complexity       | Simple HashMap-based               | No advanced data structures     |
//!
//! ## When to Use
//!
//! **Use when:**
//! - Implementing a database buffer pool where scan resistance is critical
//! - Cost of cache miss (disk I/O) >> CPU cost of O(1) list maintenance
//! - Cache size is moderate, or evictions are infrequent vs. hits
//!
//! **Avoid when:**
//! - You need exact LRU-K semantics (this is an O(1) approximation)
//! - Cache size is very large (millions of items) with frequent evictions
//! - High-frequency, low-latency environment (e.g., CPU cache simulation)
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! use crate::storage::disk::async_disk::cache::lru_k::LRUKCache;
//! use crate::storage::disk::async_disk::cache::cache_traits::{
//!     CoreCache, MutableCache, LRUKCacheTrait,
//! };
//!
//! // Create LRU-2 cache (default K=2)
//! let mut cache: LRUKCache<u32, String> = LRUKCache::new(100);
//!
//! // Or with custom K value
//! let mut cache: LRUKCache<u32, String> = LRUKCache::with_k(100, 3);
//!
//! // Insert items
//! cache.insert(1, "page_data_1".to_string());
//! cache.insert(2, "page_data_2".to_string());
//!
//! // Access items (updates history)
//! if let Some(value) = cache.get(&1) {
//!     println!("Got: {}", value);
//! }
//!
//! // Check access count
//! assert_eq!(cache.access_count(&1), Some(2)); // insert + get
//!
//! // Touch without retrieving (useful for pinned pages)
//! cache.touch(&1);
//! assert_eq!(cache.access_count(&1), Some(3));
//!
//! // Check K-distance (None if < K accesses)
//! if let Some(k_dist) = cache.k_distance(&1) {
//!     println!("K-distance: {} microseconds", k_dist);
//! }
//!
//! // Get access history (most recent first)
//! if let Some(history) = cache.access_history(&1) {
//!     println!("Access times: {:?}", history);
//! }
//!
//! // Peek at eviction victim without removing
//! if let Some((key, value)) = cache.peek_lru_k() {
//!     println!("Next victim: key={}, value={}", key, value);
//! }
//!
//! // Manually evict
//! if let Some((key, value)) = cache.pop_lru_k() {
//!     println!("Evicted: key={}, value={}", key, value);
//! }
//!
//! // Check eviction priority rank (0 = first to be evicted)
//! if let Some(rank) = cache.k_distance_rank(&2) {
//!     println!("Eviction rank: {}", rank);
//! }
//! ```
//!
//! ## Comparison with Other Policies
//!
//! | Policy | K-distance | Scan Resistant | Eviction | Best For                |
//! |--------|------------|----------------|----------|-------------------------|
//! | LRU    | K=1        | No             | O(1)     | Simple recency patterns |
//! | LRU-2  | K=2        | Yes            | O(1)     | DB buffer pools         |
//! | LRU-K  | Any K      | Yes            | O(1)     | Tunable scan resistance |
//! | LFU    | Frequency  | Partial        | O(log N) | Frequency-heavy loads   |
//!
//! ## Thread Safety
//!
//! - `LRUKCache` is **NOT thread-safe**
//! - Wrap in `Mutex` or `RwLock` for concurrent access
//! - Or use single-threaded context
//!
//! ## Academic Reference
//!
//! O'Neil, E. J., O'Neil, P. E., & Weikum, G. (1993).
//! "The LRU-K page replacement algorithm for database disk buffering."
//! ACM SIGMOD Record, 22(2), 297-306.

use std::collections::{HashMap, VecDeque};
use std::hash::Hash;

#[cfg(feature = "metrics")]
use crate::metrics::metrics_impl::LruKMetrics;
#[cfg(feature = "metrics")]
use crate::metrics::snapshot::LruKMetricsSnapshot;
#[cfg(feature = "metrics")]
use crate::metrics::traits::{
    CoreMetricsRecorder, LruKMetricsReadRecorder, LruKMetricsRecorder, LruMetricsRecorder,
    MetricsSnapshotProvider,
};
use crate::traits::{CoreCache, LRUKCacheTrait, MutableCache};

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum Segment {
    Cold,
    Hot,
}

#[derive(Debug)]
struct Entry<K, V> {
    key: K,
    value: V,
    history: VecDeque<u64>,
    segment: Segment,
}

#[derive(Debug)]
struct Slot<K, V> {
    entry: Option<Entry<K, V>>,
    prev: Option<usize>,
    next: Option<usize>,
}

#[derive(Debug, Default)]
struct LruList {
    head: Option<usize>,
    tail: Option<usize>,
    len: usize,
}

impl LruList {
    fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// LRU-K Cache implementation.
///
/// This cache evicts the item whose K-th most recent access is furthest in the past.
#[derive(Debug)]
pub struct LRUKCache<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    capacity: usize,
    k: usize,
    slots: Vec<Slot<K, V>>,
    free_list: Vec<usize>,
    index: HashMap<K, usize>,
    cold: LruList,
    hot: LruList,
    tick: u64,
    #[cfg(feature = "metrics")]
    metrics: LruKMetrics,
}

impl<K, V> LRUKCache<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    /// Creates a new LRU-K cache with default K=2.
    pub fn new(capacity: usize) -> Self {
        Self::with_k(capacity, 2)
    }

    /// Creates a new LRU-K cache with the specified capacity and K value.
    pub fn with_k(capacity: usize, k: usize) -> Self {
        let k = k.max(1);
        LRUKCache {
            capacity,
            k,
            slots: Vec::with_capacity(capacity),
            free_list: Vec::new(),
            index: HashMap::with_capacity(capacity),
            cold: LruList::default(),
            hot: LruList::default(),
            tick: 0,
            #[cfg(feature = "metrics")]
            metrics: LruKMetrics::default(),
        }
    }

    fn allocate_slot(&mut self, entry: Entry<K, V>) -> usize {
        if let Some(idx) = self.free_list.pop() {
            self.slots[idx] = Slot {
                entry: Some(entry),
                prev: None,
                next: None,
            };
            idx
        } else {
            self.slots.push(Slot {
                entry: Some(entry),
                prev: None,
                next: None,
            });
            self.slots.len() - 1
        }
    }

    fn remove_slot(&mut self, idx: usize) -> Entry<K, V> {
        let entry = self.slots[idx].entry.take().expect("lru-k entry missing");
        self.slots[idx].prev = None;
        self.slots[idx].next = None;
        self.free_list.push(idx);
        entry
    }

    fn list_push_front(slots: &mut [Slot<K, V>], list: &mut LruList, idx: usize) {
        let old_head = list.head;
        slots[idx].prev = None;
        slots[idx].next = old_head;
        if let Some(head_idx) = old_head {
            slots[head_idx].prev = Some(idx);
        } else {
            list.tail = Some(idx);
        }
        list.head = Some(idx);
        list.len += 1;
    }

    fn list_remove(slots: &mut [Slot<K, V>], list: &mut LruList, idx: usize) {
        let prev = slots[idx].prev;
        let next = slots[idx].next;
        if let Some(prev_idx) = prev {
            slots[prev_idx].next = next;
        } else {
            list.head = next;
        }
        if let Some(next_idx) = next {
            slots[next_idx].prev = prev;
        } else {
            list.tail = prev;
        }
        slots[idx].prev = None;
        slots[idx].next = None;
        list.len = list.len.saturating_sub(1);
    }

    fn list_pop_back(slots: &mut [Slot<K, V>], list: &mut LruList) -> Option<usize> {
        let idx = list.tail?;
        Self::list_remove(slots, list, idx);
        Some(idx)
    }

    fn record_access(&mut self, idx: usize) -> usize {
        self.tick = self.tick.saturating_add(1);
        let entry = self.slots[idx].entry.as_mut().expect("lru-k entry missing");
        entry.history.push_back(self.tick);
        if entry.history.len() > self.k {
            entry.history.pop_front();
        }
        entry.history.len()
    }

    fn promote_if_needed(&mut self, idx: usize) {
        let promote = self
            .slots
            .get(idx)
            .and_then(|slot| slot.entry.as_ref())
            .map(|entry| entry.segment == Segment::Cold && entry.history.len() >= self.k)
            .unwrap_or(false);
        if !promote {
            return;
        }

        let slots = &mut self.slots;
        Self::list_remove(slots, &mut self.cold, idx);

        if let Some(entry) = slots[idx].entry.as_mut() {
            entry.segment = Segment::Hot;
        }
        Self::list_push_front(slots, &mut self.hot, idx);
    }

    fn move_hot_to_front(&mut self, idx: usize) {
        let is_hot = self
            .slots
            .get(idx)
            .and_then(|slot| slot.entry.as_ref())
            .map(|entry| entry.segment == Segment::Hot)
            .unwrap_or(false);
        if !is_hot {
            return;
        }
        let slots = &mut self.slots;
        Self::list_remove(slots, &mut self.hot, idx);
        Self::list_push_front(slots, &mut self.hot, idx);
    }

    fn evict_candidate(&mut self) -> Option<(K, V)> {
        let slots = &mut self.slots;
        let idx = if !self.cold.is_empty() {
            Self::list_pop_back(slots, &mut self.cold)?
        } else {
            Self::list_pop_back(slots, &mut self.hot)?
        };

        let entry = self.remove_slot(idx);
        self.index.remove(&entry.key);
        Some((entry.key, entry.value))
    }

    fn peek_candidate(&self) -> Option<usize> {
        if !self.cold.is_empty() {
            self.cold.tail
        } else {
            self.hot.tail
        }
    }
}

// Implementation of the new specialized traits
impl<K, V> CoreCache<K, V> for LRUKCache<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    fn insert(&mut self, key: K, value: V) -> Option<V> {
        #[cfg(feature = "metrics")]
        self.metrics.record_insert_call();

        if self.capacity == 0 {
            return None;
        }

        if let Some(&idx) = self.index.get(&key) {
            #[cfg(feature = "metrics")]
            self.metrics.record_insert_update();

            let old_value = {
                let entry = self.slots[idx].entry.as_mut().expect("lru-k entry missing");
                std::mem::replace(&mut entry.value, value)
            };

            self.record_access(idx);
            self.promote_if_needed(idx);
            self.move_hot_to_front(idx);

            return Some(old_value);
        }

        #[cfg(feature = "metrics")]
        self.metrics.record_insert_new();

        if self.index.len() >= self.capacity && !self.index.is_empty() {
            #[cfg(feature = "metrics")]
            self.metrics.record_evict_call();

            if let Some((_key, _value)) = self.evict_candidate() {
                #[cfg(feature = "metrics")]
                self.metrics.record_evicted_entry();
            }
        }

        self.tick = self.tick.saturating_add(1);
        let mut history = VecDeque::with_capacity(self.k);
        history.push_back(self.tick);
        let entry = Entry {
            key: key.clone(),
            value,
            history,
            segment: Segment::Cold,
        };
        let idx = self.allocate_slot(entry);
        self.index.insert(key, idx);
        Self::list_push_front(&mut self.slots, &mut self.cold, idx);

        None
    }

    fn get(&mut self, key: &K) -> Option<&V> {
        let idx = match self.index.get(key) {
            Some(idx) => *idx,
            None => {
                #[cfg(feature = "metrics")]
                self.metrics.record_get_miss();
                return None;
            },
        };

        self.record_access(idx);
        self.promote_if_needed(idx);
        self.move_hot_to_front(idx);

        #[cfg(feature = "metrics")]
        self.metrics.record_get_hit();

        self.slots[idx].entry.as_ref().map(|entry| &entry.value)
    }

    fn contains(&self, key: &K) -> bool {
        self.index.contains_key(key)
    }

    fn len(&self) -> usize {
        self.index.len()
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn clear(&mut self) {
        #[cfg(feature = "metrics")]
        self.metrics.record_clear();
        self.index.clear();
        self.slots.clear();
        self.free_list.clear();
        self.cold = LruList::default();
        self.hot = LruList::default();
        self.tick = 0;
    }
}

impl<K, V> MutableCache<K, V> for LRUKCache<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    fn remove(&mut self, key: &K) -> Option<V> {
        let idx = self.index.remove(key)?;
        let segment = self.slots[idx]
            .entry
            .as_ref()
            .expect("lru-k entry missing")
            .segment;
        if segment == Segment::Cold {
            Self::list_remove(&mut self.slots, &mut self.cold, idx);
        } else {
            Self::list_remove(&mut self.slots, &mut self.hot, idx);
        }
        let entry = self.remove_slot(idx);
        Some(entry.value)
    }
}

impl<K, V> LRUKCacheTrait<K, V> for LRUKCache<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    fn pop_lru_k(&mut self) -> Option<(K, V)> {
        #[cfg(feature = "metrics")]
        self.metrics.record_pop_lru_k_call();

        let result = self.evict_candidate();

        #[cfg(feature = "metrics")]
        if result.is_some() {
            self.metrics.record_pop_lru_k_found();
        }

        result
    }

    fn peek_lru_k(&self) -> Option<(&K, &V)> {
        #[cfg(feature = "metrics")]
        (&self.metrics).record_peek_lru_k_call();

        let idx = self.peek_candidate()?;
        let entry = self.slots[idx].entry.as_ref()?;

        #[cfg(feature = "metrics")]
        (&self.metrics).record_peek_lru_k_found();

        Some((&entry.key, &entry.value))
    }

    fn k_value(&self) -> usize {
        self.k
    }

    fn access_history(&self, key: &K) -> Option<Vec<u64>> {
        let idx = self.index.get(key)?;
        self.slots[*idx].entry.as_ref().map(|entry| {
            entry.history.iter().rev().copied().collect() // Most recent first
        })
    }

    fn access_count(&self, key: &K) -> Option<usize> {
        let idx = self.index.get(key)?;
        self.slots[*idx]
            .entry
            .as_ref()
            .map(|entry| entry.history.len())
    }

    fn k_distance(&self, key: &K) -> Option<u64> {
        #[cfg(feature = "metrics")]
        (&self.metrics).record_k_distance_call();

        let result = self
            .index
            .get(key)
            .and_then(|idx| self.slots[*idx].entry.as_ref())
            .and_then(|entry| {
                if entry.history.len() >= self.k {
                    entry.history.front().copied()
                } else {
                    None
                }
            });

        #[cfg(feature = "metrics")]
        if result.is_some() {
            (&self.metrics).record_k_distance_found();
        }

        result
    }

    fn touch(&mut self, key: &K) -> bool {
        #[cfg(feature = "metrics")]
        self.metrics.record_touch_call();

        let idx = match self.index.get(key) {
            Some(idx) => *idx,
            None => return false,
        };
        self.record_access(idx);
        self.promote_if_needed(idx);
        self.move_hot_to_front(idx);

        #[cfg(feature = "metrics")]
        self.metrics.record_touch_found();
        true
    }

    fn k_distance_rank(&self, key: &K) -> Option<usize> {
        #[cfg(feature = "metrics")]
        (&self.metrics).record_k_distance_rank_call();

        if !self.index.contains_key(key) {
            return None;
        }

        let mut items_with_distances: Vec<(bool, u64)> = Vec::new();

        for idx in self.index.values() {
            let history = &self.slots[*idx]
                .entry
                .as_ref()
                .expect("lru-k entry missing")
                .history;
            #[cfg(feature = "metrics")]
            (&self.metrics).record_k_distance_rank_scan_step();

            let num_accesses = history.len();

            if num_accesses < self.k {
                // Items with fewer than K accesses use their earliest access time
                let earliest = history.front().copied().unwrap_or(u64::MAX);
                items_with_distances.push((false, earliest)); // false = not full K accesses
            } else {
                // Items with K or more accesses use their K-distance
                let k_distance = history.front().copied().unwrap_or(u64::MAX);
                items_with_distances.push((true, k_distance)); // true = has full K accesses
            }
        }

        // Sort by priority: items with fewer than K accesses first (by earliest access),
        // then items with K+ accesses (by K-distance)
        items_with_distances.sort_by(|a, b| {
            match (a.0, b.0) {
                (false, false) => a.1.cmp(&b.1), // Both have < K accesses, sort by earliest
                (true, true) => a.1.cmp(&b.1),   // Both have >= K accesses, sort by K-distance
                (false, true) => std::cmp::Ordering::Less, // < K accesses comes first
                (true, false) => std::cmp::Ordering::Greater, // >= K accesses comes second
            }
        });

        // Find the rank of the target key
        let target_idx = *self.index.get(key)?;
        let target_history = &self.slots[target_idx]
            .entry
            .as_ref()
            .expect("lru-k entry missing")
            .history;
        let target_num_accesses = target_history.len();
        let target_value = if target_num_accesses < self.k {
            (false, target_history.front().copied().unwrap_or(u64::MAX))
        } else {
            (true, target_history.front().copied().unwrap_or(u64::MAX))
        };

        items_with_distances
            .iter()
            .position(|item| item == &target_value)
            .inspect(|_| {
                #[cfg(feature = "metrics")]
                (&self.metrics).record_k_distance_rank_found();
            })
    }
}

#[cfg(feature = "metrics")]
impl<K, V> LRUKCache<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    pub fn metrics_snapshot(&self) -> LruKMetricsSnapshot {
        LruKMetricsSnapshot {
            get_calls: self.metrics.get_calls,
            get_hits: self.metrics.get_hits,
            get_misses: self.metrics.get_misses,
            insert_calls: self.metrics.insert_calls,
            insert_updates: self.metrics.insert_updates,
            insert_new: self.metrics.insert_new,
            evict_calls: self.metrics.evict_calls,
            evicted_entries: self.metrics.evicted_entries,
            pop_lru_calls: self.metrics.pop_lru_calls,
            pop_lru_found: self.metrics.pop_lru_found,
            peek_lru_calls: self.metrics.peek_lru_calls,
            peek_lru_found: self.metrics.peek_lru_found,
            touch_calls: self.metrics.touch_calls,
            touch_found: self.metrics.touch_found,
            recency_rank_calls: self.metrics.recency_rank_calls,
            recency_rank_found: self.metrics.recency_rank_found,
            recency_rank_scan_steps: self.metrics.recency_rank_scan_steps,
            pop_lru_k_calls: self.metrics.pop_lru_k_calls,
            pop_lru_k_found: self.metrics.pop_lru_k_found,
            peek_lru_k_calls: self.metrics.peek_lru_k_calls.get(),
            peek_lru_k_found: self.metrics.peek_lru_k_found.get(),
            k_distance_calls: self.metrics.k_distance_calls.get(),
            k_distance_found: self.metrics.k_distance_found.get(),
            k_distance_rank_calls: self.metrics.k_distance_rank_calls.get(),
            k_distance_rank_found: self.metrics.k_distance_rank_found.get(),
            k_distance_rank_scan_steps: self.metrics.k_distance_rank_scan_steps.get(),
            cache_len: self.index.len(),
            capacity: self.capacity,
        }
    }
}

#[cfg(feature = "metrics")]
impl<K, V> MetricsSnapshotProvider<LruKMetricsSnapshot> for LRUKCache<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    fn snapshot(&self) -> LruKMetricsSnapshot {
        self.metrics_snapshot()
    }
}

#[cfg(test)]
mod tests {
    mod basic_behavior {
        use std::thread;
        use std::time::Duration;

        use super::super::*;

        #[test]
        fn test_basic_lru_k_insertion_and_retrieval() {
            let mut cache = LRUKCache::new(2);
            cache.insert(1, "one");
            assert_eq!(cache.get(&1), Some(&"one"));

            cache.insert(2, "two");
            assert_eq!(cache.get(&2), Some(&"two"));
            assert_eq!(cache.len(), 2);
        }

        #[test]
        fn test_lru_k_eviction_order() {
            // Capacity 3, K=2
            let mut cache = LRUKCache::with_k(3, 2);

            // Access pattern:
            // 1: access (history: [t1]) -> < K accesses
            cache.insert(1, 10);
            thread::sleep(Duration::from_millis(2));

            // 2: access (history: [t2]) -> < K accesses
            cache.insert(2, 20);
            thread::sleep(Duration::from_millis(2));

            // 3: access (history: [t3]) -> < K accesses
            cache.insert(3, 30);
            thread::sleep(Duration::from_millis(2));

            // Cache full: {1, 2, 3} all have 1 access.
            // Eviction policy prioritizes items with < K accesses, then earliest access.
            // 1 is oldest.

            // Insert 4
            cache.insert(4, 40);

            assert!(!cache.contains(&1), "1 should be evicted");
            assert!(cache.contains(&2));
            assert!(cache.contains(&3));
            assert!(cache.contains(&4));

            // Now make 2 have K accesses.
            thread::sleep(Duration::from_millis(2));
            cache.get(&2); // 2 now has 2 accesses.

            // Current state:
            // 2: 2 accesses (>= K). Last access t5.
            // 3: 1 access (< K). Last access t3.
            // 4: 1 access (< K). Last access t4.

            // If we insert 5, we look for items with < K accesses first.
            // Candidates: 3, 4.
            // 3 is older (t3 < t4). Victim: 3.

            cache.insert(5, 50);
            assert!(!cache.contains(&3), "3 should be evicted");
            assert!(cache.contains(&2));
            assert!(cache.contains(&4));
            assert!(cache.contains(&5));
        }

        #[test]
        fn test_capacity_enforcement() {
            let mut cache = LRUKCache::new(2);
            cache.insert(1, 1);
            thread::sleep(Duration::from_millis(1));
            cache.insert(2, 2);
            assert_eq!(cache.len(), 2);

            cache.insert(3, 3);
            assert_eq!(cache.len(), 2);
            // 1 should be evicted (earliest access, < K)
            assert!(!cache.contains(&1));
            assert!(cache.contains(&2));
            assert!(cache.contains(&3));
        }

        #[test]
        fn test_update_existing_key() {
            let mut cache = LRUKCache::new(2);
            cache.insert(1, 10);
            cache.insert(1, 20);

            assert_eq!(cache.get(&1), Some(&20));
            // Insert counts as access. First insert = 1 access. Second insert = 2 accesses.
            assert_eq!(cache.access_count(&1), Some(2));
        }

        #[test]
        fn test_access_history_tracking() {
            let mut cache = LRUKCache::with_k(2, 3); // K=3
            cache.insert(1, 10); // 1 access
            thread::sleep(Duration::from_millis(2));

            cache.get(&1); // 2 accesses
            thread::sleep(Duration::from_millis(2));

            cache.get(&1); // 3 accesses
            thread::sleep(Duration::from_millis(2));

            assert_eq!(cache.access_count(&1), Some(3));

            cache.get(&1); // 4 accesses. Should keep last 3.
            assert_eq!(cache.access_count(&1), Some(3));

            let history = cache.access_history(&1).unwrap();
            assert_eq!(history.len(), 3);
            // Verify order (most recent first)
            assert!(history[0] > history[1]);
            assert!(history[1] > history[2]);
        }

        #[test]
        fn test_k_value_behavior() {
            let cache = LRUKCache::<i32, i32>::with_k(10, 5);
            assert_eq!(cache.k_value(), 5);
        }

        #[test]
        fn test_key_operations_consistency() {
            let mut cache = LRUKCache::new(2);
            cache.insert(1, 10);

            assert!(cache.contains(&1));
            assert_eq!(cache.get(&1), Some(&10));
            assert_eq!(cache.len(), 1);
        }

        #[test]
        fn test_timestamp_ordering() {
            let mut cache = LRUKCache::with_k(2, 1); // K=1 (LRU)
            cache.insert(1, 10);
            thread::sleep(Duration::from_millis(2));
            cache.insert(2, 20);

            let dist1 = cache.k_distance(&1).unwrap();
            let dist2 = cache.k_distance(&2).unwrap();

            // K=1, k_distance is the timestamp of the last access.
            // 2 was inserted after 1, so dist2 > dist1.
            assert!(dist2 > dist1);
        }
    }

    // Edge Cases Tests
    mod edge_cases {
        use std::thread;
        use std::time::Duration;

        use super::super::*;

        #[test]
        fn test_empty_cache_operations() {
            let mut cache = LRUKCache::<i32, i32>::new(5);
            assert_eq!(cache.get(&1), None);
            assert_eq!(cache.remove(&1), None);
            assert_eq!(cache.len(), 0);
            assert!(!cache.contains(&1));
        }

        #[test]
        fn test_single_item_cache() {
            let mut cache = LRUKCache::new(1);
            cache.insert(1, 10);
            assert_eq!(cache.len(), 1);
            assert_eq!(cache.get(&1), Some(&10));

            cache.insert(2, 20);
            assert_eq!(cache.len(), 1);
            assert_eq!(cache.get(&2), Some(&20));
            assert!(!cache.contains(&1));
        }

        #[test]
        fn test_zero_capacity_cache() {
            let mut cache = LRUKCache::new(0);
            cache.insert(1, 10);
            assert_eq!(cache.len(), 0);
            assert!(!cache.contains(&1));
        }

        #[test]
        fn test_k_equals_one() {
            // K=1 behaves like regular LRU
            let mut cache = LRUKCache::with_k(2, 1);

            cache.insert(1, 10);
            thread::sleep(Duration::from_millis(2));

            cache.insert(2, 20);
            thread::sleep(Duration::from_millis(2));

            // Access 1 to make it most recent
            cache.get(&1);
            thread::sleep(Duration::from_millis(2));

            // Cache: 1 (MRU), 2 (LRU)
            cache.insert(3, 30);

            assert!(cache.contains(&1));
            assert!(!cache.contains(&2)); // 2 was LRU
            assert!(cache.contains(&3));
        }

        #[test]
        fn test_k_larger_than_capacity() {
            let mut cache = LRUKCache::with_k(2, 5); // K=5, Cap=2

            cache.insert(1, 10);
            thread::sleep(Duration::from_millis(1)); // Ensure t1 < t2
            cache.insert(2, 20);

            // Access them a few times
            cache.get(&1);
            cache.get(&2);

            // Both have < K accesses. Eviction based on earliest access.
            // 1 was inserted first, then accessed.
            // 2 was inserted second, then accessed.
            // Timestamps:
            // 1: t1, t3
            // 2: t2, t4
            // Earliest access for 1 is t1. Earliest access for 2 is t2.
            // t1 < t2. So 1 should be evicted if we strictly follow "earliest access" rule for < K.

            cache.insert(3, 30);
            assert!(!cache.contains(&1));
            assert!(cache.contains(&2));
            assert!(cache.contains(&3));
        }

        #[test]
        fn test_same_key_rapid_accesses() {
            let mut cache = LRUKCache::with_k(5, 3);
            cache.insert(1, 10);
            for _ in 0..10 {
                cache.get(&1);
            }
            assert_eq!(cache.access_count(&1), Some(3)); // History capped at K
        }

        #[test]
        fn test_duplicate_key_insertion() {
            let mut cache = LRUKCache::new(5);
            cache.insert(1, 10);
            cache.insert(1, 20);
            assert_eq!(cache.get(&1), Some(&20));
            assert_eq!(cache.len(), 1);
        }

        #[test]
        fn test_large_cache_operations() {
            let mut cache = LRUKCache::new(100);

            // Insert 0 first and wait to ensure it has the distinctly oldest timestamp
            cache.insert(0, 0);
            thread::sleep(Duration::from_millis(1));

            for i in 1..100 {
                cache.insert(i, i);
            }
            assert_eq!(cache.len(), 100);

            cache.insert(100, 100);
            assert_eq!(cache.len(), 100);
            assert!(!cache.contains(&0)); // 0 should be evicted (oldest, < K)
        }

        #[test]
        fn test_access_history_overflow() {
            let mut cache = LRUKCache::with_k(2, 3); // K=3
            cache.insert(1, 10);
            cache.get(&1);
            cache.get(&1);
            cache.get(&1);
            cache.get(&1);

            let history = cache.access_history(&1).unwrap();
            assert_eq!(history.len(), 3);
        }
    }

    // LRU-K-Specific Operations Tests
    mod lru_k_operations {
        use std::thread;
        use std::time::Duration;

        use super::super::*;

        #[test]
        fn test_pop_lru_k_basic() {
            let mut cache = LRUKCache::with_k(3, 2);
            cache.insert(1, 10);
            thread::sleep(Duration::from_millis(2));
            cache.insert(2, 20);

            // Both < K accesses. 1 is older.
            let popped = cache.pop_lru_k();
            assert_eq!(popped, Some((1, 10)));
            assert!(!cache.contains(&1));
            assert_eq!(cache.len(), 1);
        }

        #[test]
        fn test_peek_lru_k_basic() {
            let mut cache = LRUKCache::with_k(3, 2);
            cache.insert(1, 10);
            thread::sleep(Duration::from_millis(2));
            cache.insert(2, 20);

            // 1 should be the victim
            let peeked = cache.peek_lru_k();
            assert_eq!(peeked, Some((&1, &10)));
            assert!(cache.contains(&1));
            assert_eq!(cache.len(), 2);
        }

        #[test]
        fn test_k_value_retrieval() {
            let cache = LRUKCache::<i32, i32>::with_k(10, 4);
            assert_eq!(cache.k_value(), 4);
        }

        #[test]
        fn test_access_history_retrieval() {
            let mut cache = LRUKCache::with_k(10, 3);
            cache.insert(1, 10);
            cache.get(&1);

            let history = cache.access_history(&1).unwrap();
            assert_eq!(history.len(), 2);
            // Check if history is returned
        }

        #[test]
        fn test_access_count() {
            let mut cache = LRUKCache::new(5);
            cache.insert(1, 10);
            assert_eq!(cache.access_count(&1), Some(1));
            cache.get(&1);
            assert_eq!(cache.access_count(&1), Some(2));
        }

        #[test]
        fn test_k_distance() {
            let mut cache = LRUKCache::with_k(5, 2);
            cache.insert(1, 10);

            // < K accesses, k_distance returns None
            assert_eq!(cache.k_distance(&1), None);

            cache.get(&1);
            // >= K accesses, returns Some(timestamp)
            assert!(cache.k_distance(&1).is_some());
        }

        #[test]
        fn test_touch_functionality() {
            let mut cache = LRUKCache::new(5);
            cache.insert(1, 10);

            assert!(cache.touch(&1));
            assert_eq!(cache.access_count(&1), Some(2));

            assert!(!cache.touch(&999)); // Non-existent key
        }

        #[test]
        fn test_k_distance_rank() {
            let mut cache = LRUKCache::with_k(5, 2);

            cache.insert(1, 10); // < K
            thread::sleep(Duration::from_millis(2));
            cache.insert(2, 20); // < K
            thread::sleep(Duration::from_millis(2));

            // 1 is oldest < K. Rank should be 0 (most eligible for eviction).
            // 2 is newer < K. Rank should be 1.

            // The method `k_distance_rank` logic:
            // Sorts by: (< K accesses, earliest access), then (>= K accesses, K-distance).
            // Returns index in this sorted list.

            // 1: < K, t1
            // 2: < K, t2
            // t1 < t2, so 1 comes first.

            assert_eq!(cache.k_distance_rank(&1), Some(0));
            assert_eq!(cache.k_distance_rank(&2), Some(1));

            cache.get(&1); // 1 now has >= K (2 accesses).
            // 1: >= K, t1 (k-dist is t1? No, k-dist is k-th most recent access).
            // History for 1: [t1, t3]. K=2. k-th most recent is t1.
            // 2: < K, t2.

            // List sorted:
            // (< K items first): 2 (t2)
            // (>= K items next): 1 (t1)

            // So 2 should be rank 0. 1 should be rank 1.
            assert_eq!(cache.k_distance_rank(&2), Some(0));
            assert_eq!(cache.k_distance_rank(&1), Some(1));
        }

        #[test]
        fn test_pop_lru_k_empty_cache() {
            let mut cache = LRUKCache::<i32, i32>::new(5);
            assert_eq!(cache.pop_lru_k(), None);
        }

        #[test]
        fn test_peek_lru_k_empty_cache() {
            let cache = LRUKCache::<i32, i32>::new(5);
            assert_eq!(cache.peek_lru_k(), None);
        }

        #[test]
        fn test_lru_k_tie_breaking() {
            let mut cache = LRUKCache::with_k(5, 2);
            // Since we use a monotonic logical clock, true ties are unlikely without mocking.
            // But logic says: if same K-distance, result is undefined/implementation dependent
            // unless we have secondary sort key.
            // The implementation handles "same number of accesses" for < K by checking earliest access.
            // For >= K, it just compares K-distance.
            // If K-distances are equal, it picks one (the first one encountered or last).

            // We can test that it returns *something*.
            cache.insert(1, 10);
            cache.insert(2, 20);
            // Both < K.
            assert!(cache.peek_lru_k().is_some());
        }

        #[test]
        fn test_access_history_after_removal() {
            let mut cache = LRUKCache::new(5);
            cache.insert(1, 10);
            cache.remove(&1);

            assert!(!cache.contains(&1));
            assert_eq!(cache.access_count(&1), None);
        }

        #[test]
        fn test_access_history_after_clear() {
            let mut cache = LRUKCache::new(5);
            cache.insert(1, 10);
            cache.clear();

            assert_eq!(cache.len(), 0);
            assert_eq!(cache.access_count(&1), None);
        }
    }

    // State Consistency Tests
    mod state_consistency {
        use super::super::*;

        #[test]
        fn test_cache_access_history_consistency() {
            let mut cache = LRUKCache::new(5);
            cache.insert(1, 10);

            // Check if access history exists for inserted key
            assert!(cache.access_history(&1).is_some());

            // Check if access history is removed for removed key
            cache.remove(&1);
            assert!(cache.access_history(&1).is_none());
        }

        #[test]
        fn test_len_consistency() {
            let mut cache = LRUKCache::new(5);
            assert_eq!(cache.len(), 0);

            cache.insert(1, 10);
            assert_eq!(cache.len(), 1);

            cache.insert(2, 20);
            assert_eq!(cache.len(), 2);

            cache.remove(&1);
            assert_eq!(cache.len(), 1);

            cache.clear();
            assert_eq!(cache.len(), 0);
        }

        #[test]
        fn test_capacity_consistency() {
            let mut cache = LRUKCache::new(2);
            assert_eq!(cache.capacity(), 2);

            cache.insert(1, 10);
            cache.insert(2, 20);
            cache.insert(3, 30);

            assert_eq!(cache.len(), 2); // Should not exceed capacity
        }

        #[test]
        fn test_clear_resets_all_state() {
            let mut cache = LRUKCache::new(5);
            cache.insert(1, 10);
            cache.insert(2, 20);

            cache.clear();

            assert_eq!(cache.len(), 0);
            assert!(!cache.contains(&1));
            assert!(!cache.contains(&2));
            assert!(cache.access_history(&1).is_none());
        }

        #[test]
        fn test_remove_consistency() {
            let mut cache = LRUKCache::new(5);
            cache.insert(1, 10);

            let removed = cache.remove(&1);
            assert_eq!(removed, Some(10));
            assert!(!cache.contains(&1));
            assert!(cache.access_history(&1).is_none());

            let removed_again = cache.remove(&1);
            assert_eq!(removed_again, None);
        }

        #[test]
        fn test_eviction_consistency() {
            let mut cache = LRUKCache::new(1);
            cache.insert(1, 10);

            // Should evict 1
            cache.insert(2, 20);

            assert!(!cache.contains(&1));
            assert!(cache.contains(&2));
            assert!(cache.access_history(&1).is_none());
            assert!(cache.access_history(&2).is_some());
        }

        #[test]
        fn test_access_history_update_on_get() {
            let mut cache = LRUKCache::new(5);
            cache.insert(1, 10);

            let count_before = cache.access_count(&1).unwrap();
            cache.get(&1);
            let count_after = cache.access_count(&1).unwrap();

            assert_eq!(count_after, count_before + 1);
        }

        #[test]
        fn test_invariants_after_operations() {
            let mut cache = LRUKCache::with_k(2, 2);
            cache.insert(1, 10);
            cache.insert(2, 20);

            // Invariant: len <= capacity
            assert!(cache.len() <= cache.capacity());

            // Invariant: history length <= K
            let h1 = cache.access_history(&1).unwrap();
            assert!(h1.len() <= 2);

            cache.get(&1);
            cache.get(&1);
            let h1_new = cache.access_history(&1).unwrap();
            assert!(h1_new.len() <= 2);
        }

        #[test]
        fn test_k_distance_calculation_consistency() {
            let mut cache = LRUKCache::with_k(5, 2);
            cache.insert(1, 10); // 1 access

            assert_eq!(cache.k_distance(&1), None);

            cache.get(&1); // 2 accesses
            assert!(cache.k_distance(&1).is_some());
        }

        #[test]
        fn test_timestamp_consistency() {
            let mut cache = LRUKCache::new(5);
            cache.insert(1, 10);

            let history = cache.access_history(&1).unwrap();
            let ts1 = history[0];

            std::thread::sleep(std::time::Duration::from_millis(1));
            cache.get(&1);

            let history_new = cache.access_history(&1).unwrap();
            let ts2 = history_new[0]; // Most recent

            assert!(ts2 > ts1);
        }
    }
}
