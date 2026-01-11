//! # LFU (Least Frequently Used) Cache Implementation
//!
//! This module provides a production-ready LFU cache implementation designed for Ferrite's
//! storage layer. The LFU cache evicts the least frequently accessed items when capacity
//! is reached, making it ideal for workloads with stable access patterns.
//!
//! ## Architecture
//!
//! ```text
//!   ┌──────────────────────────────────────────────────────────────────────────┐
//!   │                          LFUCache<K, V>                                  │
//!   │                                                                          │
//!   │   ┌────────────────────────────────────────────────────────────────────┐ │
//!   │   │  HashMap<K, usize> + Slot<K> (freq + list links)                   │ │
//!   │   │                                                                    │ │
//!   │   │  ┌─────────┬───────────────────────────────────────────────────┐   │ │
//!   │   │  │   Key   │  (Frequency)                                      │   │ │
//!   │   │  ├─────────┼───────────────────────────────────────────────────┤   │ │
//!   │   │  │ page_1  │  15  ← Hot: accessed frequently                   │   │ │
//!   │   │  │ page_2  │   3  ← Warm: moderate accesses                    │   │ │
//!   │   │  │ page_3  │   1  ← Cold: single access (LFU victim)           │   │ │
//!   │   │  │ page_4  │   7  ← Warm: moderate accesses                    │   │ │
//!   │   │  └─────────┴───────────────────────────────────────────────────┘   │ │
//!   │   │                                                                    │ │
//!   │   │  Eviction: O(1) bucket pop using min_freq                           │ │
//!   │   └────────────────────────────────────────────────────────────────────┘ │
//!   │                                                                          │
//!   │   ┌────────────────────────────────────────────────────────────────────┐ │
//!   │   │  HashMapStore<K, V> (values live here)                             │ │
//!   │   │  K -> Arc<V>                                                       │ │
//!   │   └────────────────────────────────────────────────────────────────────┘ │
//!   │                                                                          │
//!   │   capacity: usize  (maximum entries)                                     │
//!   └──────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## LFU vs LRU Comparison
//!
//! ```text
//!   Access pattern: A, B, A, C, A, D, A, E, A, F  (A accessed 5 times, others 1 each)
//!   Cache capacity: 3
//!
//!   LRU (recency-based):
//!   ═══════════════════════════════════════════════════════════════════════════
//!     After A,B,A,C: [A, C, B]  (most recent → least recent)
//!     Insert D:      [D, A, C]  ← B evicted (least recent)
//!     Insert E:      [E, D, A]  ← C evicted
//!     Insert F:      [F, E, D]  ← A evicted! (even though accessed 5 times)
//!
//!   LFU (frequency-based):
//!   ═══════════════════════════════════════════════════════════════════════════
//!     After A,B,A,C: {A:3, B:1, C:1}
//!     Insert D:      {A:3, D:1, C:1}  ← B evicted (freq=1, arbitrary tie-break)
//!     Insert E:      {A:5, E:1, D:1}  ← C evicted (freq=1)
//!     Insert F:      {A:5, F:1, E:1}  ← D evicted (freq=1)
//!
//!   Result: A (hot item) survives in LFU, evicted in LRU!
//! ```
//!
//! ## Eviction Flow
//!
//! ```text
//!   insert(new_key, new_value)
//!        │
//!        ▼
//!   ┌────────────────────────────────────────────────────────────────────────┐
//!   │ Key already exists?                                                    │
//!   │                                                                        │
//!   │   YES → Update value, preserve frequency, return old value             │
//!   │   NO  → Continue to capacity check                                     │
//!   └────────────────────────────────────────────────────────────────────────┘
//!        │
//!        ▼
//!   ┌────────────────────────────────────────────────────────────────────────┐
//!   │ Cache at capacity?                                                     │
//!   │                                                                        │
//!   │   NO  → Insert new entry with frequency = 1                            │
//!   │   YES → Find and evict LFU item (O(1) bucket pop)                      │
//!   └────────────────────────────────────────────────────────────────────────┘
//!        │
//!        ▼ (capacity reached)
//!   ┌────────────────────────────────────────────────────────────────────────┐
//!   │ LFU Eviction (O(1)):                                                   │
//!   │                                                                        │
//!   │   1. Use min_freq to select the lowest bucket                          │
//!   │   2. Pop the LRU entry in that bucket                                  │
//!   │   3. Insert new entry with frequency = 1                               │
//!   │                                                                        │
//!   │   Tie-breaking: LRU order within the lowest-frequency bucket           │
//!   └────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Frequency Lifecycle
//!
//! ```text
//!   insert(key, value)
//!        │
//!        ▼
//!   ┌─────────────────┐
//!   │ Frequency = 1   │  ← Initial state (cold item)
//!   └─────────────────┘
//!        │
//!        │ get(&key), increment_frequency(&key)
//!        ▼
//!   ┌─────────────────┐
//!   │ Frequency += 1  │  ← Each access increments
//!   └─────────────────┘
//!        │
//!        │ reset_frequency(&key)
//!        ▼
//!   ┌─────────────────┐
//!   │ Frequency = 1   │  ← Manual reset (for aging)
//!   └─────────────────┘
//!        │
//!        │ remove(&key), pop_lfu(), clear()
//!        ▼
//!   ┌─────────────────┐
//!   │ Entry removed   │  ← Frequency tracking gone
//!   └─────────────────┘
//! ```
//!
//! ## Key Components
//!
//! | Component        | Description                                        |
//! |------------------|----------------------------------------------------|
//! | `LFUCache<K, V>` | Main cache struct                                  |
//! | `index`          | `HashMap<K, usize>` to slot indices                |
//! | `freq_buckets`   | Per-frequency LRU buckets + links                  |
//! | `store`          | Stores key -> `Arc<V>` ownership                   |
//!
//! ## Core Operations (CoreCache + MutableCache)
//!
//! | Method           | Complexity | Description                              |
//! |------------------|------------|------------------------------------------|
//! | `new(capacity)`  | O(1)       | Create cache with given capacity         |
//! | `insert(k, v)`   | O(1)*      | Insert `Arc<V>`, may trigger O(1) eviction |
//! | `get(&k)`        | O(1)       | Get value, increments frequency          |
//! | `contains(&k)`   | O(1)       | Check if key exists                      |
//! | `remove(&k)`     | O(1)       | Remove entry by key                      |
//! | `len()`          | O(1)       | Current number of entries                |
//! | `capacity()`     | O(1)       | Maximum capacity                         |
//! | `clear()`        | O(n)       | Remove all entries                       |
//!
//! ## LFU-Specific Operations (LFUCacheTrait)
//!
//! | Method                   | Complexity | Description                       |
//! |--------------------------|------------|-----------------------------------|
//! | `pop_lfu()`              | O(1)       | Remove and return LFU item        |
//! | `peek_lfu()`             | O(1)       | Peek at LFU item without removing |
//! | `frequency(&k)`          | O(1)       | Get frequency count for key       |
//! | `reset_frequency(&k)`    | O(1)       | Reset frequency to 1              |
//! | `increment_frequency(&k)`| O(1)       | Manually increment frequency      |
//!
//! ## Performance Characteristics
//!
//! | Operation              | Time       | Notes                              |
//! |------------------------|------------|------------------------------------|
//! | `get`                  | O(1)       | Index lookup + freq increment      |
//! | `insert` (no eviction) | O(1)       | Index insert + store insert        |
//! | `insert` (eviction)    | O(1)       | Bucket pop via min_freq            |
//! | `pop_lfu`              | O(1)       | Bucket pop via min_freq            |
//! | `peek_lfu`             | O(1)       | Tail lookup in min_freq bucket     |
//! | Per-entry overhead     | ~24 bytes  | Key + freq + index + store entry   |
//!
//! ## Trade-offs
//!
//! | Aspect           | Pros                              | Cons                            |
//! |------------------|-----------------------------------|---------------------------------|
//! | Hot Item Retain  | Keeps frequently accessed items   | Cold start problem              |
//! | Eviction Quality | Good for stable access patterns   | O(1) eviction                    |
//! | Memory           | Store + index, simple structure   | No frequency decay/aging        |
//! | Simplicity       | Easy to understand and debug      | Non-deterministic tie-breaking  |
//!
//! ## Limitations
//!
//! 1. **Bucketed LFU**: `pop_lfu()` and `peek_lfu()` use the min_freq bucket
//! 2. **Cold Start Problem**: New items have frequency 1, easily evicted
//! 3. **No Aging**: Old frequent items stay forever unless manually reset
//! 4. **Tie-Breaking**: Non-deterministic when multiple items have same frequency
//! 5. **Not Thread-Safe**: Requires external synchronization
//!
//! ## When to Use
//!
//! **Use when:**
//! - Database buffer pools with stable access patterns
//! - Computational caches with expensive-to-recompute results
//! - Reference data (lookup tables, dictionaries)
//! - Analytical workloads identifying hot data
//!
//! **Avoid when:**
//! - Temporal locality dominates (use LRU)
//! - Frequent `pop_lfu()`/`peek_lfu()` calls needed (use heap-based LFU)
//! - Access patterns shift rapidly (consider adaptive policies)
//! - Real-time systems requiring bounded O(1) latency
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! use crate::storage::disk::async_disk::cache::lfu::LFUCache;
//! use std::sync::Arc;
//! use crate::storage::disk::async_disk::cache::cache_traits::{
//!     CoreCache, MutableCache, LFUCacheTrait,
//! };
//!
//! // Create cache
//! let mut cache: LFUCache<String, i32> = LFUCache::new(100);
//!
//! // Insert items (frequency starts at 1)
//! cache.insert("key1".to_string(), Arc::new(100));
//! cache.insert("key2".to_string(), Arc::new(200));
//!
//! // Access increments frequency
//! cache.get(&"key1".to_string()); // freq: 1 → 2
//! cache.get(&"key1".to_string()); // freq: 2 → 3
//!
//! assert_eq!(cache.frequency(&"key1".to_string()), Some(3));
//! assert_eq!(cache.frequency(&"key2".to_string()), Some(1));
//!
//! // Manual frequency control
//! cache.increment_frequency(&"key2".to_string()); // freq: 1 → 2
//! cache.reset_frequency(&"key1".to_string());     // freq: 3 → 1
//!
//! // Peek at LFU candidate (O(1) bucket pop)
//! if let Some((key, value)) = cache.peek_lfu() {
//!     println!("Next victim: {} = {}", key, value.as_ref());
//! }
//!
//! // Evict LFU item (O(1) bucket pop)
//! if let Some((key, value)) = cache.pop_lfu() {
//!     println!("Evicted: {} = {}", key, value.as_ref());
//! }
//!
//! // Thread-safe usage
//! use std::sync::{Arc, Mutex};
//! let shared_cache = Arc::new(Mutex::new(LFUCache::<u64, Vec<u8>>::new(1000)));
//!
//! // In thread:
//! {
//!     let mut cache = shared_cache.lock().unwrap();
//!     cache.insert(page_id, Arc::new(page_data));
//! }
//! ```
//!
//! ## Comparison with Other Policies
//!
//! | Policy   | Eviction Basis | Eviction Time | Best For                  |
//! |----------|----------------|---------------|---------------------------|
//! | LFU      | Frequency      | O(1)          | Stable access patterns    |
//! | LRU      | Recency        | O(1)          | Temporal locality         |
//! | LRU-K    | K-th access    | O(1)          | Scan resistance           |
//! | FIFO     | Insertion time | O(1)          | Simple, predictable       |
//!
//! ## Thread Safety
//!
//! - `LFUCache` is **NOT thread-safe**
//! - Wrap in `Arc<Mutex<LFUCache>>` or `Arc<RwLock<LFUCache>>` for concurrent access
//! - Note: Long critical sections still matter; keep list operations tight
//!
//! ## Implementation Notes
//!
//! - **Key Clone Requirement**: Keys must be `Clone` for O(1) indexing
//! - **Zero Capacity**: Supported - rejects all insertions
//! - **Frequency Overflow**: Theoretically possible at `usize::MAX` accesses
//! - **Store + Index**: Values live in the store; slots track frequency and order

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;

#[cfg(feature = "metrics")]
use crate::metrics::metrics_impl::LfuMetrics;
#[cfg(feature = "metrics")]
use crate::metrics::snapshot::LfuMetricsSnapshot;
#[cfg(feature = "metrics")]
use crate::metrics::traits::{
    CoreMetricsRecorder, LfuMetricsReadRecorder, LfuMetricsRecorder, MetricsSnapshotProvider,
};
use crate::store::hashmap::HashMapStore;
use crate::store::traits::{StoreCore, StoreMut};
use crate::traits::{CoreCache, LFUCacheTrait, MutableCache};

#[derive(Debug)]
struct Entry<K> {
    key: K,
    freq: u64,
}

#[derive(Debug)]
struct Slot<K> {
    entry: Option<Entry<K>>,
    prev: Option<usize>,
    next: Option<usize>,
}

#[derive(Debug, Default)]
struct FreqList {
    head: Option<usize>,
    tail: Option<usize>,
    len: usize,
}

impl FreqList {
    fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[derive(Debug, Default)]
struct FreqBucket {
    list: FreqList,
    prev: Option<u64>,
    next: Option<u64>,
}

/// LFU (Least Frequently Used) Cache.
///
/// Evicts the item with the lowest access frequency when capacity is reached.
/// See module-level documentation for details.
#[derive(Debug)]
pub struct LFUCache<K, V>
where
    K: Eq + Hash + Clone,
{
    store: HashMapStore<K, V>,
    slots: Vec<Slot<K>>,
    free_list: Vec<usize>,
    index: HashMap<K, usize>,
    freq_buckets: HashMap<u64, FreqBucket>,
    min_freq: u64,
    #[cfg(feature = "metrics")]
    metrics: LfuMetrics,
}

impl<K, V> LFUCache<K, V>
where
    K: Eq + Hash + Clone,
{
    pub fn new(capacity: usize) -> Self {
        LFUCache {
            store: HashMapStore::new(capacity),
            slots: Vec::with_capacity(capacity),
            free_list: Vec::new(),
            index: HashMap::with_capacity(capacity),
            freq_buckets: HashMap::new(),
            min_freq: 0,
            #[cfg(feature = "metrics")]
            metrics: LfuMetrics::default(),
        }
    }

    fn allocate_slot(&mut self, entry: Entry<K>) -> usize {
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

    fn remove_slot(&mut self, idx: usize) -> Entry<K> {
        let entry = self.slots[idx].entry.take().expect("lfu entry missing");
        self.slots[idx].prev = None;
        self.slots[idx].next = None;
        self.free_list.push(idx);
        entry
    }

    fn list_push_front(slots: &mut [Slot<K>], list: &mut FreqList, idx: usize) {
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

    fn list_remove(slots: &mut [Slot<K>], list: &mut FreqList, idx: usize) {
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

    fn list_pop_back(slots: &mut [Slot<K>], list: &mut FreqList) -> Option<usize> {
        let idx = list.tail?;
        Self::list_remove(slots, list, idx);
        Some(idx)
    }

    fn insert_bucket(&mut self, freq: u64, prev: Option<u64>, next: Option<u64>) {
        self.freq_buckets.insert(
            freq,
            FreqBucket {
                list: FreqList::default(),
                prev,
                next,
            },
        );

        if let Some(prev_freq) = prev
            && let Some(bucket) = self.freq_buckets.get_mut(&prev_freq)
        {
            bucket.next = Some(freq);
        }

        if let Some(next_freq) = next
            && let Some(bucket) = self.freq_buckets.get_mut(&next_freq)
        {
            bucket.prev = Some(freq);
        }

        if prev.is_none() {
            self.min_freq = freq;
        }
    }

    fn move_to_freq(&mut self, idx: usize, new_freq: u64) {
        let old_freq = self
            .slots
            .get(idx)
            .and_then(|slot| slot.entry.as_ref())
            .expect("lfu entry missing")
            .freq;

        if old_freq == new_freq {
            return;
        }

        let (old_prev, old_next, old_bucket_empty) = {
            let slots = &mut self.slots;
            if let Some(bucket) = self.freq_buckets.get_mut(&old_freq) {
                let prev = bucket.prev;
                let next = bucket.next;
                Self::list_remove(slots, &mut bucket.list, idx);
                (prev, next, bucket.list.is_empty())
            } else {
                (None, None, false)
            }
        };

        if old_bucket_empty {
            self.freq_buckets.remove(&old_freq);
            if let Some(prev_freq) = old_prev
                && let Some(bucket) = self.freq_buckets.get_mut(&prev_freq)
            {
                bucket.next = old_next;
            }
            if let Some(next_freq) = old_next
                && let Some(bucket) = self.freq_buckets.get_mut(&next_freq)
            {
                bucket.prev = old_prev;
            }
            if self.min_freq == old_freq {
                self.min_freq = old_next.unwrap_or(0);
            }
        }

        if let Some(entry) = self.slots[idx].entry.as_mut() {
            entry.freq = new_freq;
        }
        if self.freq_buckets.contains_key(&new_freq) {
            let bucket = self
                .freq_buckets
                .get_mut(&new_freq)
                .expect("lfu bucket missing");
            Self::list_push_front(&mut self.slots, &mut bucket.list, idx);
        } else {
            let (prev, next) = if new_freq == 1 {
                (
                    None,
                    if self.min_freq == 0 {
                        None
                    } else {
                        Some(self.min_freq)
                    },
                )
            } else if new_freq > old_freq {
                let prev = if old_bucket_empty {
                    old_prev
                } else {
                    Some(old_freq)
                };
                let next = old_next;
                (prev, next)
            } else {
                (
                    None,
                    if self.min_freq == 0 {
                        None
                    } else {
                        Some(self.min_freq)
                    },
                )
            };

            self.insert_bucket(new_freq, prev, next);
            let bucket = self
                .freq_buckets
                .get_mut(&new_freq)
                .expect("lfu bucket missing");
            Self::list_push_front(&mut self.slots, &mut bucket.list, idx);
        }

        if self.min_freq == 0 || new_freq < self.min_freq {
            self.min_freq = new_freq;
        }
    }

    fn evict_min_freq(&mut self) -> Option<(K, Arc<V>)> {
        let min_freq = self.min_freq;
        if min_freq == 0 {
            return None;
        }

        let (idx, next_freq, list_empty) = {
            let slots = &mut self.slots;
            let bucket = self.freq_buckets.get_mut(&min_freq)?;
            let idx = Self::list_pop_back(slots, &mut bucket.list)?;
            (idx, bucket.next, bucket.list.is_empty())
        };

        if list_empty {
            let prev = self
                .freq_buckets
                .get(&min_freq)
                .and_then(|bucket| bucket.prev);
            self.freq_buckets.remove(&min_freq);
            if let Some(prev_freq) = prev
                && let Some(bucket) = self.freq_buckets.get_mut(&prev_freq)
            {
                bucket.next = next_freq;
            }
            if let Some(next_freq) = next_freq
                && let Some(bucket) = self.freq_buckets.get_mut(&next_freq)
            {
                bucket.prev = prev;
            }
            if self.min_freq == min_freq {
                self.min_freq = next_freq.unwrap_or(0);
            }
        }

        let entry = self.remove_slot(idx);
        self.index.remove(&entry.key);
        self.store.record_eviction();
        let value = self.store.remove(&entry.key)?;
        Some((entry.key, value))
    }
}

// Implementation of specialized traits
impl<K, V> CoreCache<K, Arc<V>> for LFUCache<K, V>
where
    K: Eq + Hash + Clone,
{
    fn insert(&mut self, key: K, value: Arc<V>) -> Option<Arc<V>> {
        #[cfg(feature = "metrics")]
        self.metrics.record_insert_call();

        if self.index.contains_key(&key) {
            #[cfg(feature = "metrics")]
            self.metrics.record_insert_update();

            return self.store.try_insert(key, value).ok().flatten();
        }

        // Handle zero capacity case - reject all new insertions
        if self.store.capacity() == 0 {
            return None;
        }

        #[cfg(feature = "metrics")]
        self.metrics.record_insert_new();

        if self.index.len() >= self.store.capacity() {
            #[cfg(feature = "metrics")]
            self.metrics.record_evict_call();

            if let Some((_key, _value)) = self.evict_min_freq() {
                #[cfg(feature = "metrics")]
                self.metrics.record_evicted_entry();
            }
        }

        if self.store.try_insert(key.clone(), value).is_err() {
            return None;
        }

        let entry = Entry {
            key: key.clone(),
            freq: 1,
        };
        let idx = self.allocate_slot(entry);
        self.index.insert(key, idx);
        if !self.freq_buckets.contains_key(&1) {
            let next = if self.min_freq == 0 {
                None
            } else {
                Some(self.min_freq)
            };
            self.insert_bucket(1, None, next);
        }
        let bucket = self.freq_buckets.get_mut(&1).expect("lfu bucket missing");
        Self::list_push_front(&mut self.slots, &mut bucket.list, idx);
        self.min_freq = 1;

        None
    }

    fn get(&mut self, key: &K) -> Option<&Arc<V>> {
        let idx = match self.index.get(key) {
            Some(idx) => *idx,
            None => {
                #[cfg(feature = "metrics")]
                self.metrics.record_get_miss();
                let _ = self.store.get_ref(key);
                return None;
            },
        };

        let new_freq = self.slots[idx]
            .entry
            .as_ref()
            .expect("lfu entry missing")
            .freq
            .saturating_add(1);
        self.move_to_freq(idx, new_freq);

        #[cfg(feature = "metrics")]
        self.metrics.record_get_hit();

        self.store.get_ref(key)
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
        #[cfg(feature = "metrics")]
        self.metrics.record_clear();
        self.store.clear();
        self.index.clear();
        self.freq_buckets.clear();
        self.slots.clear();
        self.free_list.clear();
        self.min_freq = 0;
    }
}

impl<K, V> MutableCache<K, Arc<V>> for LFUCache<K, V>
where
    K: Eq + Hash + Clone,
{
    fn remove(&mut self, key: &K) -> Option<Arc<V>> {
        let idx = self.index.remove(key)?;
        let freq = self.slots[idx]
            .entry
            .as_ref()
            .expect("lfu entry missing")
            .freq;

        let (list_empty, next_freq, prev_freq) =
            if let Some(bucket) = self.freq_buckets.get_mut(&freq) {
                let prev = bucket.prev;
                let next = bucket.next;
                Self::list_remove(&mut self.slots, &mut bucket.list, idx);
                (bucket.list.is_empty(), next, prev)
            } else {
                (false, None, None)
            };

        if list_empty {
            self.freq_buckets.remove(&freq);
            if let Some(prev) = prev_freq
                && let Some(bucket) = self.freq_buckets.get_mut(&prev)
            {
                bucket.next = next_freq;
            }
            if let Some(next) = next_freq
                && let Some(bucket) = self.freq_buckets.get_mut(&next)
            {
                bucket.prev = prev_freq;
            }
            if self.min_freq == freq {
                self.min_freq = next_freq.unwrap_or(0);
            }
        }

        let entry = self.remove_slot(idx);
        self.store.remove(&entry.key)
    }
}

impl<K, V> LFUCacheTrait<K, Arc<V>> for LFUCache<K, V>
where
    K: Eq + Hash + Clone,
{
    fn pop_lfu(&mut self) -> Option<(K, Arc<V>)> {
        #[cfg(feature = "metrics")]
        self.metrics.record_pop_lfu_call();

        let result = self.evict_min_freq();

        #[cfg(feature = "metrics")]
        if result.is_some() {
            self.metrics.record_pop_lfu_found();
        }

        result
    }

    fn peek_lfu(&self) -> Option<(&K, &Arc<V>)> {
        #[cfg(feature = "metrics")]
        (&self.metrics).record_peek_lfu_call();

        let min_freq = self.min_freq;
        let idx = self
            .freq_buckets
            .get(&min_freq)
            .and_then(|bucket| bucket.list.tail)?;
        let entry = self.slots[idx].entry.as_ref()?;
        let value = self.store.peek_ref(&entry.key)?;

        #[cfg(feature = "metrics")]
        (&self.metrics).record_peek_lfu_found();

        Some((&entry.key, value))
    }

    fn frequency(&self, key: &K) -> Option<u64> {
        #[cfg(feature = "metrics")]
        (&self.metrics).record_frequency_call();

        let result = self
            .index
            .get(key)
            .and_then(|idx| self.slots[*idx].entry.as_ref())
            .map(|entry| entry.freq);

        #[cfg(feature = "metrics")]
        if result.is_some() {
            (&self.metrics).record_frequency_found();
        }

        result
    }

    fn reset_frequency(&mut self, key: &K) -> Option<u64> {
        #[cfg(feature = "metrics")]
        self.metrics.record_reset_frequency_call();

        let idx = *self.index.get(key)?;
        let previous_freq = self.slots[idx]
            .entry
            .as_ref()
            .expect("lfu entry missing")
            .freq;
        self.move_to_freq(idx, 1);

        #[cfg(feature = "metrics")]
        self.metrics.record_reset_frequency_found();

        Some(previous_freq)
    }

    fn increment_frequency(&mut self, key: &K) -> Option<u64> {
        #[cfg(feature = "metrics")]
        self.metrics.record_increment_frequency_call();

        let idx = *self.index.get(key)?;
        let new_freq = self.slots[idx]
            .entry
            .as_ref()
            .expect("lfu entry missing")
            .freq
            .saturating_add(1);
        self.move_to_freq(idx, new_freq);

        #[cfg(feature = "metrics")]
        self.metrics.record_increment_frequency_found();

        Some(new_freq)
    }
}

#[cfg(feature = "metrics")]
impl<K, V> LFUCache<K, V>
where
    K: Eq + Hash + Clone,
{
    pub fn metrics_snapshot(&self) -> LfuMetricsSnapshot {
        LfuMetricsSnapshot {
            get_calls: self.metrics.get_calls,
            get_hits: self.metrics.get_hits,
            get_misses: self.metrics.get_misses,
            insert_calls: self.metrics.insert_calls,
            insert_updates: self.metrics.insert_updates,
            insert_new: self.metrics.insert_new,
            evict_calls: self.metrics.evict_calls,
            evicted_entries: self.metrics.evicted_entries,
            pop_lfu_calls: self.metrics.pop_lfu_calls,
            pop_lfu_found: self.metrics.pop_lfu_found,
            peek_lfu_calls: self.metrics.peek_lfu_calls.get(),
            peek_lfu_found: self.metrics.peek_lfu_found.get(),
            frequency_calls: self.metrics.frequency_calls.get(),
            frequency_found: self.metrics.frequency_found.get(),
            reset_frequency_calls: self.metrics.reset_frequency_calls,
            reset_frequency_found: self.metrics.reset_frequency_found,
            increment_frequency_calls: self.metrics.increment_frequency_calls,
            increment_frequency_found: self.metrics.increment_frequency_found,
            cache_len: self.store.len(),
            capacity: self.store.capacity(),
        }
    }

    #[cfg(debug_assertions)]
    #[cfg(test)]
    pub(crate) fn debug_validate_invariants(&self) {
        assert!(self.len() <= self.capacity());
        assert_eq!(self.len(), self.index.len());

        if self.len() == 0 {
            assert_eq!(self.min_freq, 0);
            assert!(self.freq_buckets.is_empty());
            return;
        }

        assert!(self.min_freq > 0);
        assert!(self.freq_buckets.contains_key(&self.min_freq));

        for (&freq, bucket) in &self.freq_buckets {
            assert!(!bucket.list.is_empty());
            if let Some(prev) = bucket.prev {
                assert!(self.freq_buckets.contains_key(&prev));
                assert_eq!(self.freq_buckets[&prev].next, Some(freq));
            } else {
                assert_eq!(self.min_freq, freq);
            }
            if let Some(next) = bucket.next {
                assert!(self.freq_buckets.contains_key(&next));
                assert_eq!(self.freq_buckets[&next].prev, Some(freq));
            }

            let mut current = bucket.list.head;
            let mut last = None;
            let mut count = 0usize;
            while let Some(idx) = current {
                let slot = self.slots.get(idx).expect("lfu slot missing");
                let entry = slot.entry.as_ref().expect("lfu entry missing");
                assert_eq!(entry.freq, freq);
                assert_eq!(slot.prev, last);
                last = Some(idx);
                current = slot.next;
                count += 1;
            }
            assert_eq!(bucket.list.tail, last);
            assert_eq!(bucket.list.len, count);
        }
    }
}

#[cfg(feature = "metrics")]
impl<K, V> MetricsSnapshotProvider<LfuMetricsSnapshot> for LFUCache<K, V>
where
    K: Eq + Hash + Clone,
{
    fn snapshot(&self) -> LfuMetricsSnapshot {
        self.metrics_snapshot()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::CoreCache;

    // Basic LFU Behavior Tests
    mod basic_behavior {
        use super::*;

        #[test]
        fn test_basic_lfu_insertion_and_retrieval() {
            let mut cache = LFUCache::new(3);

            // Test insertion and basic retrieval
            assert_eq!(cache.insert("key1".to_string(), Arc::new(100)), None);
            assert_eq!(cache.insert("key2".to_string(), Arc::new(200)), None);
            assert_eq!(cache.insert("key3".to_string(), Arc::new(300)), None);

            // Test retrieval
            assert_eq!(cache.get(&"key1".to_string()).map(Arc::as_ref), Some(&100));
            assert_eq!(cache.get(&"key2".to_string()).map(Arc::as_ref), Some(&200));
            assert_eq!(cache.get(&"key3".to_string()).map(Arc::as_ref), Some(&300));

            // Test non-existent key
            assert_eq!(cache.get(&"nonexistent".to_string()), None);

            // Test that initial frequencies are 1, then increment on access
            assert_eq!(cache.frequency(&"key1".to_string()), Some(2)); // 1 + 1 from get
            assert_eq!(cache.frequency(&"key2".to_string()), Some(2)); // 1 + 1 from get
            assert_eq!(cache.frequency(&"key3".to_string()), Some(2)); // 1 + 1 from get
        }

        #[test]
        fn test_lfu_eviction_order() {
            let mut cache = LFUCache::new(3);

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
        fn test_capacity_enforcement() {
            let mut cache = LFUCache::new(2);

            // Verify initial state
            assert_eq!(cache.len(), 0);
            assert_eq!(cache.capacity(), 2);

            // Insert first item
            cache.insert("key1".to_string(), Arc::new(100));
            assert_eq!(cache.len(), 1);
            assert!(cache.len() <= cache.capacity());

            // Insert second item (at capacity)
            cache.insert("key2".to_string(), Arc::new(200));
            assert_eq!(cache.len(), 2);
            assert!(cache.len() <= cache.capacity());

            // Insert third item (should trigger eviction)
            cache.insert("key3".to_string(), Arc::new(300));
            assert_eq!(cache.len(), 2); // Should still be 2
            assert!(cache.len() <= cache.capacity());

            // Insert many more items
            for i in 4..=10 {
                cache.insert(format!("key{}", i), Arc::new(i * 100));
                assert!(cache.len() <= cache.capacity());
                assert_eq!(cache.len(), 2);
            }

            // Test with zero capacity
            let mut zero_cache = LFUCache::new(0);
            assert_eq!(zero_cache.capacity(), 0);
            assert_eq!(zero_cache.len(), 0);

            // Insert into zero capacity cache
            zero_cache.insert("key".to_string(), Arc::new(100));
            assert_eq!(zero_cache.len(), 0); // Should remain 0
            assert!(zero_cache.len() <= zero_cache.capacity());
        }

        #[test]
        fn test_update_existing_key() {
            let mut cache = LFUCache::new(3);

            // Insert initial value
            assert_eq!(cache.insert("key1".to_string(), Arc::new(100)), None);
            assert_eq!(cache.frequency(&"key1".to_string()), Some(1));

            // Access the key to increase frequency
            cache.get(&"key1".to_string());
            cache.get(&"key1".to_string());
            assert_eq!(cache.frequency(&"key1".to_string()), Some(3));

            // Update the value - should preserve frequency
            let old_value = cache.insert("key1".to_string(), Arc::new(999));
            assert_eq!(old_value.as_deref(), Some(&100));
            assert_eq!(cache.frequency(&"key1".to_string()), Some(3)); // Frequency preserved

            // Verify updated value
            assert_eq!(cache.get(&"key1".to_string()).map(Arc::as_ref), Some(&999));
            assert_eq!(cache.frequency(&"key1".to_string()), Some(4)); // Incremented by get

            // Verify cache size didn't change
            assert_eq!(cache.len(), 1);

            // Add more items to test preservation during eviction scenarios
            cache.insert("key2".to_string(), Arc::new(200)); // freq = 1
            cache.insert("key3".to_string(), Arc::new(300)); // freq = 1

            // key1 has frequency 4, others have frequency 1
            // Update key1 again
            cache.insert("key1".to_string(), Arc::new(1999));
            assert_eq!(cache.frequency(&"key1".to_string()), Some(4)); // Still preserved

            // Insert new item to trigger eviction - key2 or key3 should be evicted (freq 1)
            cache.insert("key4".to_string(), Arc::new(400));

            // key1 should still be there with preserved frequency
            assert!(cache.contains(&"key1".to_string()));
            assert_eq!(cache.frequency(&"key1".to_string()), Some(4));
            assert_eq!(cache.get(&"key1".to_string()).map(Arc::as_ref), Some(&1999));
        }

        #[test]
        fn test_frequency_tracking() {
            let mut cache = LFUCache::new(5);

            // Insert items with initial frequency of 1
            cache.insert("a".to_string(), Arc::new(1));
            cache.insert("b".to_string(), Arc::new(2));
            cache.insert("c".to_string(), Arc::new(3));

            // Verify initial frequencies
            assert_eq!(cache.frequency(&"a".to_string()), Some(1));
            assert_eq!(cache.frequency(&"b".to_string()), Some(1));
            assert_eq!(cache.frequency(&"c".to_string()), Some(1));

            // Access patterns to create different frequencies
            // a: access 3 times -> freq = 4
            cache.get(&"a".to_string());
            cache.get(&"a".to_string());
            cache.get(&"a".to_string());
            assert_eq!(cache.frequency(&"a".to_string()), Some(4));

            // b: access 1 time -> freq = 2
            cache.get(&"b".to_string());
            assert_eq!(cache.frequency(&"b".to_string()), Some(2));

            // c: no additional access -> freq = 1
            assert_eq!(cache.frequency(&"c".to_string()), Some(1));

            // Test manual frequency operations
            // Reset frequency of 'a'
            let old_freq = cache.reset_frequency(&"a".to_string());
            assert_eq!(old_freq, Some(4));
            assert_eq!(cache.frequency(&"a".to_string()), Some(1));

            // Increment frequency of 'b'
            let new_freq = cache.increment_frequency(&"b".to_string());
            assert_eq!(new_freq, Some(3));
            assert_eq!(cache.frequency(&"b".to_string()), Some(3));

            // Test frequency operations on non-existent key
            assert_eq!(cache.frequency(&"nonexistent".to_string()), None);
            assert_eq!(cache.reset_frequency(&"nonexistent".to_string()), None);
            assert_eq!(cache.increment_frequency(&"nonexistent".to_string()), None);

            // Test LFU identification
            let (lfu_key, _) = cache.peek_lfu().unwrap();
            // Both 'a' and 'c' have frequency 1, so any is valid
            assert!(lfu_key == &"a".to_string() || lfu_key == &"c".to_string());

            // Verify frequency tracking after removal
            cache.remove(&"b".to_string());
            assert_eq!(cache.frequency(&"b".to_string()), None);

            // Verify frequency tracking after clear
            cache.clear();
            assert_eq!(cache.frequency(&"a".to_string()), None);
            assert_eq!(cache.frequency(&"c".to_string()), None);
            assert_eq!(cache.len(), 0);
        }

        #[test]
        fn test_key_operations_consistency() {
            let mut cache = LFUCache::new(4);

            // Test empty cache consistency
            assert_eq!(cache.len(), 0);
            assert!(!cache.contains(&"any_key".to_string()));
            assert_eq!(cache.get(&"any_key".to_string()), None);

            // Insert items and verify consistency
            let keys = vec!["key1", "key2", "key3"];
            let values = [100, 200, 300];

            for (i, (&key, &value)) in keys.iter().zip(values.iter()).enumerate() {
                cache.insert(key.to_string(), Arc::new(value));

                // Verify len is consistent
                assert_eq!(cache.len(), i + 1);

                // Verify contains is consistent with successful insertion
                assert!(cache.contains(&key.to_string()));

                // Verify get is consistent with contains
                assert_eq!(cache.get(&key.to_string()).map(Arc::as_ref), Some(&value));
            }

            // Test consistency across all inserted keys
            for (&key, &value) in keys.iter().zip(values.iter()) {
                // contains should be true
                assert!(cache.contains(&key.to_string()));

                // get should return the value
                assert_eq!(cache.get(&key.to_string()).map(Arc::as_ref), Some(&value));

                // frequency should exist
                assert!(cache.frequency(&key.to_string()).is_some());
            }

            // Test after removal
            cache.remove(&"key2".to_string());
            assert_eq!(cache.len(), 2);
            assert!(!cache.contains(&"key2".to_string()));
            assert_eq!(cache.get(&"key2".to_string()), None);
            assert_eq!(cache.frequency(&"key2".to_string()), None);

            // Verify other keys are unaffected
            assert!(cache.contains(&"key1".to_string()));
            assert!(cache.contains(&"key3".to_string()));
            assert_eq!(cache.get(&"key1".to_string()).map(Arc::as_ref), Some(&100));
            assert_eq!(cache.get(&"key3".to_string()).map(Arc::as_ref), Some(&300));

            // Test eviction consistency
            cache.insert("key4".to_string(), Arc::new(400));
            cache.insert("key5".to_string(), Arc::new(500)); // Should trigger eviction

            assert_eq!(cache.len(), 4); // Should not exceed capacity

            // Count how many of original keys are still present
            let mut remaining_count = 0;
            for &key in &keys {
                if cache.contains(&key.to_string()) {
                    remaining_count += 1;
                    // If contains is true, get should work
                    assert!(cache.get(&key.to_string()).is_some());
                } else {
                    // If contains is false, get should return None
                    assert_eq!(cache.get(&key.to_string()), None);
                }
            }

            // At least some original keys should be evicted
            assert!(remaining_count < keys.len());

            // New keys should be present
            assert!(cache.contains(&"key4".to_string()));
            assert!(cache.contains(&"key5".to_string()));

            // Test clear consistency
            cache.clear();
            assert_eq!(cache.len(), 0);

            for &key in &["key1", "key3", "key4", "key5"] {
                assert!(!cache.contains(&key.to_string()));
                assert_eq!(cache.get(&key.to_string()), None);
                assert_eq!(cache.frequency(&key.to_string()), None);
            }
        }
    }

    // Edge Cases Tests
    mod edge_cases {
        use super::*;

        #[test]
        fn test_empty_cache_operations() {
            let mut cache = LFUCache::<String, i32>::new(5);

            // Test all operations on empty cache
            assert_eq!(cache.len(), 0);
            assert_eq!(cache.capacity(), 5);
            assert!(!cache.contains(&"nonexistent".to_string()));
            assert_eq!(cache.get(&"nonexistent".to_string()), None);
            assert_eq!(cache.frequency(&"nonexistent".to_string()), None);
            assert_eq!(cache.remove(&"nonexistent".to_string()), None);
            assert_eq!(cache.pop_lfu(), None);
            assert_eq!(cache.peek_lfu(), None);

            // Test increment/reset frequency on non-existent keys
            assert_eq!(cache.increment_frequency(&"nonexistent".to_string()), None);
            assert_eq!(cache.reset_frequency(&"nonexistent".to_string()), None);

            // Clear empty cache should work
            cache.clear();
            assert_eq!(cache.len(), 0);
        }

        #[test]
        fn test_single_item_cache() {
            let mut cache = LFUCache::new(1);

            // Test initial state
            assert_eq!(cache.len(), 0);
            assert_eq!(cache.capacity(), 1);

            // Insert first item
            assert_eq!(cache.insert("key1".to_string(), Arc::new(100)), None);
            assert_eq!(cache.len(), 1);
            assert!(cache.contains(&"key1".to_string()));
            assert_eq!(cache.get(&"key1".to_string()).map(Arc::as_ref), Some(&100));
            assert_eq!(cache.frequency(&"key1".to_string()), Some(2)); // 1 from insert + 1 from get

            // Insert second item should evict first
            assert_eq!(cache.insert("key2".to_string(), Arc::new(200)), None);
            assert_eq!(cache.len(), 1);
            assert!(!cache.contains(&"key1".to_string()));
            assert!(cache.contains(&"key2".to_string()));
            assert_eq!(cache.get(&"key2".to_string()).map(Arc::as_ref), Some(&200));

            // Update existing item should preserve it
            let old_value = cache.insert("key2".to_string(), Arc::new(999));
            assert_eq!(old_value.as_deref(), Some(&200));
            assert_eq!(cache.len(), 1);
            assert_eq!(cache.get(&"key2".to_string()).map(Arc::as_ref), Some(&999));

            // Test pop_lfu and peek_lfu
            assert_eq!(
                cache.peek_lfu().map(|(key, value)| (key.clone(), **value)),
                Some(("key2".to_string(), 999))
            );
            assert_eq!(
                cache.pop_lfu().map(|(key, value)| (key, *value)),
                Some(("key2".to_string(), 999))
            );
            assert_eq!(cache.len(), 0);
            assert_eq!(cache.peek_lfu(), None);
        }

        #[test]
        fn test_zero_capacity_cache() {
            let mut cache = LFUCache::<String, i32>::new(0);

            // Test initial state
            assert_eq!(cache.len(), 0);
            assert_eq!(cache.capacity(), 0);

            // All insertions should be rejected
            assert_eq!(cache.insert("key1".to_string(), Arc::new(100)), None);
            assert_eq!(cache.insert("key2".to_string(), Arc::new(200)), None);
            assert_eq!(cache.len(), 0);

            // All queries should return negative results
            assert!(!cache.contains(&"key1".to_string()));
            assert_eq!(cache.get(&"key1".to_string()), None);
            assert_eq!(cache.frequency(&"key1".to_string()), None);
            assert_eq!(cache.remove(&"key1".to_string()), None);

            // LFU operations should return None
            assert_eq!(cache.pop_lfu(), None);
            assert_eq!(cache.peek_lfu(), None);

            // Frequency operations should return None
            assert_eq!(cache.increment_frequency(&"key1".to_string()), None);
            assert_eq!(cache.reset_frequency(&"key1".to_string()), None);

            // Clear should work (no-op)
            cache.clear();
            assert_eq!(cache.len(), 0);
        }

        #[test]
        fn test_same_frequency_items() {
            let mut cache = LFUCache::new(3);

            // Insert items with same initial frequency
            cache.insert("key1".to_string(), Arc::new(100));
            cache.insert("key2".to_string(), Arc::new(200));
            cache.insert("key3".to_string(), Arc::new(300));

            // All items should have frequency 1
            assert_eq!(cache.frequency(&"key1".to_string()), Some(1));
            assert_eq!(cache.frequency(&"key2".to_string()), Some(1));
            assert_eq!(cache.frequency(&"key3".to_string()), Some(1));

            // When cache is full and we insert a new item,
            // one of the items with frequency 1 should be evicted
            let initial_keys = ["key1", "key2", "key3"];
            cache.insert("key4".to_string(), Arc::new(400));
            assert_eq!(cache.len(), 3);

            // Verify that key4 was inserted
            assert!(cache.contains(&"key4".to_string()));
            assert_eq!(cache.frequency(&"key4".to_string()), Some(1));

            // One of the original keys should be gone
            let remaining_count = initial_keys
                .iter()
                .map(|k| cache.contains(&k.to_string()))
                .filter(|&exists| exists)
                .count();
            assert_eq!(remaining_count, 2);

            // Test peek_lfu and pop_lfu behavior with same frequencies
            // Should return some item with frequency 1
            if let Some((key, _)) = cache.peek_lfu() {
                assert_eq!(cache.frequency(key), Some(1));
            }

            if let Some((key, _)) = cache.pop_lfu() {
                assert_eq!(cache.len(), 2);
                // The removed item should not be in cache anymore
                assert!(!cache.contains(&key));
            }
        }

        #[test]
        fn test_frequency_overflow_protection() {
            let mut cache = LFUCache::new(2);

            // Insert an item
            cache.insert("key1".to_string(), Arc::new(100));
            assert_eq!(cache.frequency(&"key1".to_string()), Some(1));

            // Simulate approaching overflow by setting a very high frequency
            // Since we can't directly set frequency to max, we'll test with reasonable values
            // and ensure the system doesn't panic

            // Access the item many times to increase frequency
            for _ in 0..1000 {
                cache.get(&"key1".to_string());
            }

            // Frequency should be very high but not overflow
            let freq = cache.frequency(&"key1".to_string()).unwrap();
            assert!(freq > 1000);

            // Test that increment_frequency doesn't panic with high values
            let freq_before = cache.frequency(&"key1".to_string()).unwrap();
            let freq_after_increment = cache.increment_frequency(&"key1".to_string()).unwrap();
            let freq_after = cache.frequency(&"key1".to_string()).unwrap();
            assert_eq!(freq_after_increment, freq_before + 1);
            assert_eq!(freq_after, freq_before + 1);

            // Insert another item to test that high frequency item isn't evicted
            cache.insert("key2".to_string(), Arc::new(200));
            assert_eq!(cache.len(), 2);

            // Insert third item - key2 should be evicted (lower frequency)
            cache.insert("key3".to_string(), Arc::new(300));
            assert_eq!(cache.len(), 2);
            assert!(cache.contains(&"key1".to_string())); // High frequency item preserved
            assert!(!cache.contains(&"key2".to_string())); // Low frequency item evicted
            assert!(cache.contains(&"key3".to_string())); // New item inserted
        }

        #[test]
        fn test_duplicate_key_insertion() {
            let mut cache = LFUCache::new(3);

            // Insert initial value
            assert_eq!(cache.insert("key1".to_string(), Arc::new(100)), None);
            assert_eq!(cache.len(), 1);
            assert_eq!(cache.frequency(&"key1".to_string()), Some(1));

            // Access to increase frequency
            cache.get(&"key1".to_string());
            cache.get(&"key1".to_string());
            assert_eq!(cache.frequency(&"key1".to_string()), Some(3));

            // Insert same key with different value - should update value and preserve frequency
            assert_eq!(
                cache.insert("key1".to_string(), Arc::new(999)).as_deref(),
                Some(&100)
            );
            assert_eq!(cache.len(), 1); // Length unchanged
            assert_eq!(cache.get(&"key1".to_string()).map(Arc::as_ref), Some(&999)); // Value updated
            assert_eq!(cache.frequency(&"key1".to_string()), Some(4)); // Frequency preserved + 1 for get

            // Insert again with another value
            assert_eq!(
                cache.insert("key1".to_string(), Arc::new(777)).as_deref(),
                Some(&999)
            );
            assert_eq!(cache.len(), 1);
            assert_eq!(cache.get(&"key1".to_string()).map(Arc::as_ref), Some(&777));
            assert_eq!(cache.frequency(&"key1".to_string()), Some(5)); // Frequency continues to track

            // Add other items to fill cache
            cache.insert("key2".to_string(), Arc::new(200));
            cache.insert("key3".to_string(), Arc::new(300));
            assert_eq!(cache.len(), 3);

            // Insert fourth item - key1 should not be evicted due to high frequency
            cache.insert("key4".to_string(), Arc::new(400));
            assert_eq!(cache.len(), 3);
            assert!(cache.contains(&"key1".to_string())); // High frequency item preserved

            // Verify key1 still has the correct value and frequency
            assert_eq!(cache.get(&"key1".to_string()).map(Arc::as_ref), Some(&777));

            // One of key2 or key3 should be evicted (both have frequency 1)
            let key2_exists = cache.contains(&"key2".to_string());
            let key3_exists = cache.contains(&"key3".to_string());
            assert!(!(key2_exists && key3_exists)); // Not both can exist
            assert!(cache.contains(&"key4".to_string())); // New item should exist
        }

        #[test]
        fn test_large_cache_operations() {
            let capacity = 10000;
            let mut cache = LFUCache::new(capacity);

            // Test initial state
            assert_eq!(cache.len(), 0);
            assert_eq!(cache.capacity(), capacity);

            // Insert many items
            for i in 0..capacity {
                let key = format!("key_{}", i);
                assert_eq!(cache.insert(key, Arc::new(i)), None);
            }

            // Cache should be at capacity
            assert_eq!(cache.len(), capacity);

            // All items should be present
            for i in 0..capacity {
                let key = format!("key_{}", i);
                assert!(cache.contains(&key));
                assert_eq!(cache.get(&key).map(Arc::as_ref), Some(&i));
                assert_eq!(cache.frequency(&key), Some(2)); // 1 from insert + 1 from get
            }

            // Test that additional insertion triggers eviction
            let new_key = "new_key".to_string();
            assert_eq!(cache.insert(new_key.clone(), Arc::new(99999)), None);
            assert_eq!(cache.len(), capacity); // Size should remain the same
            assert!(cache.contains(&new_key)); // New item should be present

            // Count how many original items remain (should be capacity - 1)
            let remaining_original = (0..capacity)
                .map(|i| format!("key_{}", i))
                .filter(|key| cache.contains(key))
                .count();
            assert_eq!(remaining_original, capacity - 1);

            // Test clear operation
            cache.clear();
            assert_eq!(cache.len(), 0);
            assert!(!cache.contains(&new_key));

            // Test that we can insert after clear
            cache.insert("after_clear".to_string(), Arc::new(42));
            assert_eq!(cache.len(), 1);
            assert!(cache.contains(&"after_clear".to_string()));
        }
    }

    // LFU-Specific Operations Tests
    mod lfu_operations {
        use super::*;

        #[test]
        fn test_pop_lfu_basic() {
            let mut cache = LFUCache::new(4);

            // Insert items with different access patterns
            cache.insert("key1".to_string(), Arc::new(100));
            cache.insert("key2".to_string(), Arc::new(200));
            cache.insert("key3".to_string(), Arc::new(300));

            // Create different frequencies:
            // key1: freq = 1 (no additional access)
            // key2: freq = 3 (2 additional accesses)
            // key3: freq = 2 (1 additional access)
            cache.get(&"key2".to_string());
            cache.get(&"key2".to_string());
            cache.get(&"key3".to_string());

            // Verify frequencies
            assert_eq!(cache.frequency(&"key1".to_string()), Some(1));
            assert_eq!(cache.frequency(&"key2".to_string()), Some(3));
            assert_eq!(cache.frequency(&"key3".to_string()), Some(2));

            // Pop LFU should remove key1 (lowest frequency)
            let (key, value) = cache.pop_lfu().unwrap();
            assert_eq!(key, "key1".to_string());
            assert_eq!(*value, 100);
            assert_eq!(cache.len(), 2);
            assert!(!cache.contains(&"key1".to_string()));

            // Next pop should remove key3 (next lowest frequency)
            let (key, value) = cache.pop_lfu().unwrap();
            assert_eq!(key, "key3".to_string());
            assert_eq!(*value, 300);
            assert_eq!(cache.len(), 1);

            // Final pop should remove key2
            let (key, value) = cache.pop_lfu().unwrap();
            assert_eq!(key, "key2".to_string());
            assert_eq!(*value, 200);
            assert_eq!(cache.len(), 0);
        }

        #[test]
        fn test_peek_lfu_basic() {
            let mut cache = LFUCache::new(4);

            // Insert items with different access patterns
            cache.insert("key1".to_string(), Arc::new(100));
            cache.insert("key2".to_string(), Arc::new(200));
            cache.insert("key3".to_string(), Arc::new(300));

            // Create different frequencies:
            // key1: freq = 1 (no additional access)
            // key2: freq = 3 (2 additional accesses)
            // key3: freq = 2 (1 additional access)
            cache.get(&"key2".to_string());
            cache.get(&"key2".to_string());
            cache.get(&"key3".to_string());

            // Peek LFU should return key1 (lowest frequency) without removing it
            let (key, value) = cache.peek_lfu().unwrap();
            assert_eq!(key, &"key1".to_string());
            assert_eq!(value.as_ref(), &100);
            assert_eq!(cache.len(), 3); // Cache size unchanged
            assert!(cache.contains(&"key1".to_string())); // Item still present

            // Multiple peeks should return the same result
            let (key2, value2) = cache.peek_lfu().unwrap();
            assert_eq!(key2, &"key1".to_string());
            assert_eq!(value2.as_ref(), &100);

            // After removing key1, peek should return key3 (next lowest)
            cache.remove(&"key1".to_string());
            let (key, value) = cache.peek_lfu().unwrap();
            assert_eq!(key, &"key3".to_string());
            assert_eq!(value.as_ref(), &300);
            assert_eq!(cache.len(), 2);

            // After removing key3, peek should return key2
            cache.remove(&"key3".to_string());
            let (key, value) = cache.peek_lfu().unwrap();
            assert_eq!(key, &"key2".to_string());
            assert_eq!(value.as_ref(), &200);
            assert_eq!(cache.len(), 1);
        }

        #[test]
        fn test_frequency_retrieval() {
            let mut cache = LFUCache::new(5);

            // Test frequency for non-existent key
            assert_eq!(cache.frequency(&"nonexistent".to_string()), None);

            // Insert a key and check initial frequency
            cache.insert("key1".to_string(), Arc::new(100));
            assert_eq!(cache.frequency(&"key1".to_string()), Some(1));

            // Access the key and verify frequency increments
            cache.get(&"key1".to_string());
            assert_eq!(cache.frequency(&"key1".to_string()), Some(2));

            cache.get(&"key1".to_string());
            assert_eq!(cache.frequency(&"key1".to_string()), Some(3));

            // Insert another key
            cache.insert("key2".to_string(), Arc::new(200));
            assert_eq!(cache.frequency(&"key2".to_string()), Some(1));
            assert_eq!(cache.frequency(&"key1".to_string()), Some(3)); // Unchanged

            // Access key2 multiple times
            for _ in 0..5 {
                cache.get(&"key2".to_string());
            }
            assert_eq!(cache.frequency(&"key2".to_string()), Some(6)); // 1 + 5

            // Update existing key - should preserve frequency
            cache.insert("key1".to_string(), Arc::new(999));
            assert_eq!(cache.frequency(&"key1".to_string()), Some(3)); // Preserved

            // Remove key and verify frequency is gone
            cache.remove(&"key1".to_string());
            assert_eq!(cache.frequency(&"key1".to_string()), None);
            assert_eq!(cache.frequency(&"key2".to_string()), Some(6)); // Unaffected
        }

        #[test]
        fn test_reset_frequency() {
            let mut cache = LFUCache::new(3);

            // Test reset on non-existent key
            assert_eq!(cache.reset_frequency(&"nonexistent".to_string()), None);

            // Insert a key and increase its frequency
            cache.insert("key1".to_string(), Arc::new(100));
            cache.get(&"key1".to_string());
            cache.get(&"key1".to_string());
            cache.get(&"key1".to_string());
            assert_eq!(cache.frequency(&"key1".to_string()), Some(4));

            // Reset frequency should return old frequency and set to 1
            let old_freq = cache.reset_frequency(&"key1".to_string());
            assert_eq!(old_freq, Some(4));
            assert_eq!(cache.frequency(&"key1".to_string()), Some(1));

            // Reset again should return 1
            let old_freq = cache.reset_frequency(&"key1".to_string());
            assert_eq!(old_freq, Some(1));
            assert_eq!(cache.frequency(&"key1".to_string()), Some(1));

            // Insert another key with high frequency
            cache.insert("key2".to_string(), Arc::new(200));
            for _ in 0..10 {
                cache.get(&"key2".to_string());
            }
            assert_eq!(cache.frequency(&"key2".to_string()), Some(11));

            // Reset key2 frequency
            let old_freq = cache.reset_frequency(&"key2".to_string());
            assert_eq!(old_freq, Some(11));
            assert_eq!(cache.frequency(&"key2".to_string()), Some(1));

            // Verify key1 frequency unchanged
            assert_eq!(cache.frequency(&"key1".to_string()), Some(1));

            // Test that cache still works correctly after resets
            cache.insert("key3".to_string(), Arc::new(300));
            assert_eq!(cache.len(), 3);

            // All items now have frequency 1, so eviction should be deterministic
            cache.insert("key4".to_string(), Arc::new(400)); // Should evict one of the items
            assert_eq!(cache.len(), 3);
        }

        #[test]
        fn test_increment_frequency() {
            let mut cache = LFUCache::new(3);

            // Test increment on non-existent key
            assert_eq!(cache.increment_frequency(&"nonexistent".to_string()), None);

            // Insert a key and test increment
            cache.insert("key1".to_string(), Arc::new(100));
            assert_eq!(cache.frequency(&"key1".to_string()), Some(1));

            // Increment frequency manually
            let new_freq = cache.increment_frequency(&"key1".to_string());
            assert_eq!(new_freq, Some(2));
            assert_eq!(cache.frequency(&"key1".to_string()), Some(2));

            // Increment multiple times
            for i in 3..=7 {
                let freq = cache.increment_frequency(&"key1".to_string());
                assert_eq!(freq, Some(i));
                assert_eq!(cache.frequency(&"key1".to_string()), Some(i));
            }

            // Insert another key
            cache.insert("key2".to_string(), Arc::new(200));
            assert_eq!(cache.frequency(&"key2".to_string()), Some(1));

            // Increment key2
            let freq = cache.increment_frequency(&"key2".to_string());
            assert_eq!(freq, Some(2));

            // Verify key1 frequency unchanged
            assert_eq!(cache.frequency(&"key1".to_string()), Some(7));

            // Test that increment affects LFU ordering
            cache.insert("key3".to_string(), Arc::new(300));
            assert_eq!(cache.frequency(&"key3".to_string()), Some(1));

            // key3 should be LFU (freq=1), then key2 (freq=2), then key1 (freq=7)
            let (key, _) = cache.peek_lfu().unwrap();
            assert_eq!(key, &"key3".to_string());

            // Increment key3 to make it same as key2
            cache.increment_frequency(&"key3".to_string());
            assert_eq!(cache.frequency(&"key3".to_string()), Some(2));

            // Now either key2 or key3 could be LFU (both freq=2)
            let (key, _) = cache.peek_lfu().unwrap();
            assert!(key == &"key2".to_string() || key == &"key3".to_string());
            assert_eq!(cache.frequency(key).unwrap(), 2);
        }

        #[test]
        fn test_pop_lfu_empty_cache() {
            let mut cache = LFUCache::<String, i32>::new(5);

            // Test pop_lfu on empty cache
            assert_eq!(cache.pop_lfu(), None);
            assert_eq!(cache.len(), 0);

            // Insert and remove to empty the cache again
            cache.insert("key1".to_string(), Arc::new(100));
            assert_eq!(cache.len(), 1);

            let (key, value) = cache.pop_lfu().unwrap();
            assert_eq!(key, "key1".to_string());
            assert_eq!(*value, 100);
            assert_eq!(cache.len(), 0);

            // Test pop_lfu on empty cache again
            assert_eq!(cache.pop_lfu(), None);

            // Insert multiple items and pop all
            cache.insert("a".to_string(), Arc::new(1));
            cache.insert("b".to_string(), Arc::new(2));
            cache.insert("c".to_string(), Arc::new(3));
            assert_eq!(cache.len(), 3);

            // Pop all items
            assert!(cache.pop_lfu().is_some());
            assert!(cache.pop_lfu().is_some());
            assert!(cache.pop_lfu().is_some());
            assert_eq!(cache.len(), 0);

            // Should be empty again
            assert_eq!(cache.pop_lfu(), None);
        }

        #[test]
        fn test_peek_lfu_empty_cache() {
            let cache = LFUCache::<String, i32>::new(5);

            // Test peek_lfu on empty cache
            assert_eq!(cache.peek_lfu(), None);
            assert_eq!(cache.len(), 0);

            // Test with zero capacity cache
            let zero_cache = LFUCache::<String, i32>::new(0);
            assert_eq!(zero_cache.peek_lfu(), None);
            assert_eq!(zero_cache.len(), 0);

            // Test that multiple peeks on empty cache return None
            assert_eq!(cache.peek_lfu(), None);
            assert_eq!(cache.peek_lfu(), None);
            assert_eq!(cache.peek_lfu(), None);

            // Test after creating and emptying cache
            let mut cache2 = LFUCache::new(3);
            cache2.insert("temp".to_string(), Arc::new(999));
            assert!(cache2.peek_lfu().is_some());

            cache2.clear();
            assert_eq!(cache2.peek_lfu(), None);
            assert_eq!(cache2.len(), 0);

            // Test after removing all items
            let mut cache3 = LFUCache::new(2);
            cache3.insert("a".to_string(), Arc::new(1));
            cache3.insert("b".to_string(), Arc::new(2));
            assert!(cache3.peek_lfu().is_some());

            cache3.remove(&"a".to_string());
            cache3.remove(&"b".to_string());
            assert_eq!(cache3.peek_lfu(), None);
            assert_eq!(cache3.len(), 0);
        }

        #[test]
        fn test_lfu_tie_breaking() {
            let mut cache = LFUCache::new(5);

            // Insert items and create different frequency levels
            cache.insert("low1".to_string(), Arc::new(1)); // will have freq = 1
            cache.insert("low2".to_string(), Arc::new(2)); // will have freq = 1
            cache.insert("medium".to_string(), Arc::new(3)); // will have freq = 2
            cache.insert("high".to_string(), Arc::new(4)); // will have freq = 3

            // Create frequency differences
            cache.get(&"medium".to_string()); // medium: freq = 2
            cache.get(&"high".to_string()); // high: freq = 2
            cache.get(&"high".to_string()); // high: freq = 3

            // Verify frequencies
            assert_eq!(cache.frequency(&"low1".to_string()), Some(1));
            assert_eq!(cache.frequency(&"low2".to_string()), Some(1));
            assert_eq!(cache.frequency(&"medium".to_string()), Some(2));
            assert_eq!(cache.frequency(&"high".to_string()), Some(3));

            // Test consistent tie-breaking: peek and pop should return same item
            let (peek_key, peek_value) = cache.peek_lfu().unwrap();
            let peek_key_owned = peek_key.clone();
            let peek_value_owned = **peek_value;

            let (pop_key, pop_value) = cache.pop_lfu().unwrap();
            assert_eq!(peek_key_owned, pop_key);
            assert_eq!(peek_value_owned, *pop_value);

            // The popped item should be one of the low frequency items
            assert!(pop_key == "low1" || pop_key == "low2");
            assert_eq!(cache.len(), 3);

            // Next pop should get the other low frequency item
            let (second_key, _) = cache.pop_lfu().unwrap();
            assert!(second_key == "low1" || second_key == "low2");
            assert_ne!(pop_key, second_key); // Should be different
            assert_eq!(cache.len(), 2);

            // Next should be medium frequency item
            let (third_key, third_value) = cache.pop_lfu().unwrap();
            assert_eq!(third_key, "medium".to_string());
            assert_eq!(*third_value, 3);
            assert_eq!(cache.len(), 1);

            // Finally the high frequency item
            let (last_key, last_value) = cache.pop_lfu().unwrap();
            assert_eq!(last_key, "high".to_string());
            assert_eq!(*last_value, 4);
            assert_eq!(cache.len(), 0);

            // Test with all same frequency
            cache.insert("a".to_string(), Arc::new(1));
            cache.insert("b".to_string(), Arc::new(2));
            cache.insert("c".to_string(), Arc::new(3));

            // All should have frequency 1
            assert_eq!(cache.frequency(&"a".to_string()), Some(1));
            assert_eq!(cache.frequency(&"b".to_string()), Some(1));
            assert_eq!(cache.frequency(&"c".to_string()), Some(1));

            // Should be able to pop all three (order may vary)
            let mut popped_keys = vec![
                cache.pop_lfu().unwrap().0,
                cache.pop_lfu().unwrap().0,
                cache.pop_lfu().unwrap().0,
            ];

            popped_keys.sort();
            assert_eq!(
                popped_keys,
                vec!["a".to_string(), "b".to_string(), "c".to_string()]
            );
            assert_eq!(cache.len(), 0);
        }

        #[test]
        fn test_frequency_after_removal() {
            let mut cache = LFUCache::new(5);

            // Insert items and build up frequencies
            cache.insert("key1".to_string(), Arc::new(100));
            cache.insert("key2".to_string(), Arc::new(200));
            cache.insert("key3".to_string(), Arc::new(300));

            // Increase frequencies
            for _ in 0..5 {
                cache.get(&"key1".to_string());
            }
            for _ in 0..3 {
                cache.get(&"key2".to_string());
            }
            cache.get(&"key3".to_string());

            // Verify initial frequencies
            assert_eq!(cache.frequency(&"key1".to_string()), Some(6)); // 1 + 5
            assert_eq!(cache.frequency(&"key2".to_string()), Some(4)); // 1 + 3
            assert_eq!(cache.frequency(&"key3".to_string()), Some(2)); // 1 + 1

            // Remove key1 and verify its frequency is gone
            let removed_value = cache.remove(&"key1".to_string());
            assert_eq!(removed_value.as_deref(), Some(&100));
            assert_eq!(cache.frequency(&"key1".to_string()), None);
            assert_eq!(cache.len(), 2);

            // Verify other frequencies unchanged
            assert_eq!(cache.frequency(&"key2".to_string()), Some(4));
            assert_eq!(cache.frequency(&"key3".to_string()), Some(2));

            // Test that LFU operations work correctly after removal
            let (lfu_key, _) = cache.peek_lfu().unwrap();
            assert_eq!(lfu_key, &"key3".to_string()); // Should be key3 (freq=2)

            // Remove via pop_lfu
            let (popped_key, popped_value) = cache.pop_lfu().unwrap();
            assert_eq!(popped_key, "key3".to_string());
            assert_eq!(*popped_value, 300);
            assert_eq!(cache.frequency(&"key3".to_string()), None);
            assert_eq!(cache.len(), 1);

            // Only key2 should remain
            assert_eq!(cache.frequency(&"key2".to_string()), Some(4));
            assert!(cache.contains(&"key2".to_string()));

            // Remove the last item
            cache.remove(&"key2".to_string());
            assert_eq!(cache.frequency(&"key2".to_string()), None);
            assert_eq!(cache.len(), 0);

            // Verify cache is completely empty
            assert_eq!(cache.peek_lfu(), None);
            assert_eq!(cache.pop_lfu(), None);

            // Test re-inserting with same keys creates fresh frequencies
            cache.insert("key1".to_string(), Arc::new(999));
            assert_eq!(cache.frequency(&"key1".to_string()), Some(1)); // Fresh start
        }

        #[test]
        fn test_frequency_after_clear() {
            let mut cache = LFUCache::new(5);

            // Insert items and build up frequencies
            cache.insert("key1".to_string(), Arc::new(100));
            cache.insert("key2".to_string(), Arc::new(200));
            cache.insert("key3".to_string(), Arc::new(300));

            // Increase frequencies significantly
            for _ in 0..10 {
                cache.get(&"key1".to_string());
            }
            for _ in 0..5 {
                cache.get(&"key2".to_string());
            }
            for _ in 0..7 {
                cache.get(&"key3".to_string());
            }

            // Verify high frequencies
            assert_eq!(cache.frequency(&"key1".to_string()), Some(11)); // 1 + 10
            assert_eq!(cache.frequency(&"key2".to_string()), Some(6)); // 1 + 5
            assert_eq!(cache.frequency(&"key3".to_string()), Some(8)); // 1 + 7
            assert_eq!(cache.len(), 3);

            // Clear the cache
            cache.clear();

            // Verify cache is empty
            assert_eq!(cache.len(), 0);
            assert_eq!(cache.peek_lfu(), None);
            assert_eq!(cache.pop_lfu(), None);

            // Verify all frequencies are gone
            assert_eq!(cache.frequency(&"key1".to_string()), None);
            assert_eq!(cache.frequency(&"key2".to_string()), None);
            assert_eq!(cache.frequency(&"key3".to_string()), None);

            // Verify all keys are gone
            assert!(!cache.contains(&"key1".to_string()));
            assert!(!cache.contains(&"key2".to_string()));
            assert!(!cache.contains(&"key3".to_string()));

            // Test that we can insert fresh items after clear
            cache.insert("key1".to_string(), Arc::new(999));
            cache.insert("new_key".to_string(), Arc::new(888));

            // Frequencies should start fresh
            assert_eq!(cache.frequency(&"key1".to_string()), Some(1));
            assert_eq!(cache.frequency(&"new_key".to_string()), Some(1));
            assert_eq!(cache.len(), 2);

            // Test that cache works normally after clear
            cache.get(&"key1".to_string());
            assert_eq!(cache.frequency(&"key1".to_string()), Some(2));

            // LFU operations should work
            let (lfu_key, _) = cache.peek_lfu().unwrap();
            assert_eq!(lfu_key, &"new_key".to_string()); // freq=1, should be LFU

            // Test multiple clears
            cache.clear();
            assert_eq!(cache.len(), 0);
            cache.clear(); // Should be safe to clear empty cache
            assert_eq!(cache.len(), 0);
        }

        #[test]
        fn test_bucket_link_updates_on_middle_removal() {
            let mut cache = LFUCache::new(4);

            cache.insert("low".to_string(), Arc::new(1));
            cache.insert("mid".to_string(), Arc::new(2));
            cache.insert("high".to_string(), Arc::new(3));

            cache.get(&"mid".to_string()); // mid: freq = 2
            cache.get(&"high".to_string());
            cache.get(&"high".to_string()); // high: freq = 3

            #[cfg(debug_assertions)]
            cache.debug_validate_invariants();

            cache.remove(&"mid".to_string());

            #[cfg(debug_assertions)]
            cache.debug_validate_invariants();

            let (lfu_key, _) = cache.peek_lfu().unwrap();
            assert_eq!(lfu_key, &"low".to_string());
        }
    }

    // State Consistency Tests
    mod state_consistency {
        use super::*;

        #[test]
        fn test_cache_frequency_consistency() {
            let mut cache = LFUCache::new(5);

            // Test initial state consistency
            assert_eq!(cache.len(), 0);

            // Insert items and verify frequency consistency
            cache.insert("key1".to_string(), Arc::new(100));
            cache.insert("key2".to_string(), Arc::new(200));
            cache.insert("key3".to_string(), Arc::new(300));

            // All items should have initial frequency of 1
            assert_eq!(cache.frequency(&"key1".to_string()), Some(1));
            assert_eq!(cache.frequency(&"key2".to_string()), Some(1));
            assert_eq!(cache.frequency(&"key3".to_string()), Some(1));

            // Access items to change frequencies
            cache.get(&"key1".to_string()); // key1: freq = 2
            cache.get(&"key1".to_string()); // key1: freq = 3
            cache.get(&"key2".to_string()); // key2: freq = 2

            // Verify frequency updates are consistent
            assert_eq!(cache.frequency(&"key1".to_string()), Some(3));
            assert_eq!(cache.frequency(&"key2".to_string()), Some(2));
            assert_eq!(cache.frequency(&"key3".to_string()), Some(1));

            // Test update preserves frequency
            cache.insert("key1".to_string(), Arc::new(999));
            assert_eq!(cache.frequency(&"key1".to_string()), Some(3)); // Should be preserved

            // Test manual frequency operations
            cache.increment_frequency(&"key3".to_string());
            assert_eq!(cache.frequency(&"key3".to_string()), Some(2));

            cache.reset_frequency(&"key1".to_string());
            assert_eq!(cache.frequency(&"key1".to_string()), Some(1));

            // Verify that frequency and cache remain consistent
            assert_eq!(cache.len(), 3);
            assert!(cache.contains(&"key1".to_string()));
            assert!(cache.contains(&"key2".to_string()));
            assert!(cache.contains(&"key3".to_string()));

            // Verify LFU operations use consistent frequency data
            let (lfu_key, _) = cache.peek_lfu().unwrap();
            let lfu_freq = cache.frequency(lfu_key).unwrap();
            assert_eq!(lfu_freq, 1); // Should be one of the items with frequency 1
        }

        #[test]
        fn test_len_consistency() {
            let mut cache = LFUCache::new(4);

            // Test empty cache
            assert_eq!(cache.len(), 0);

            // Test incremental insertions
            cache.insert("key1".to_string(), Arc::new(100));
            assert_eq!(cache.len(), 1);

            cache.insert("key2".to_string(), Arc::new(200));
            assert_eq!(cache.len(), 2);

            cache.insert("key3".to_string(), Arc::new(300));
            assert_eq!(cache.len(), 3);

            // Test updating existing key doesn't change length
            cache.insert("key1".to_string(), Arc::new(999));
            assert_eq!(cache.len(), 3);

            // Test insert at capacity (should increase length)
            cache.insert("key4".to_string(), Arc::new(400));
            assert_eq!(cache.len(), 4);

            // Test insert beyond capacity (should evict and maintain length)
            cache.insert("key5".to_string(), Arc::new(500));
            assert_eq!(cache.len(), 4); // Should remain at capacity

            // Test manual removals
            cache.remove(&"key5".to_string());
            assert_eq!(cache.len(), 3);

            // Eviction tie-breaking among same-frequency items is non-deterministic.
            // Remove any one of the remaining original keys that still exists.
            for candidate in ["key1", "key2", "key3", "key4"] {
                if cache.contains(&candidate.to_string()) {
                    cache.remove(&candidate.to_string());
                    break;
                }
            }
            assert_eq!(cache.len(), 2);

            // Test removing non-existent key doesn't change length
            cache.remove(&"nonexistent".to_string());
            assert_eq!(cache.len(), 2);

            // Test pop_lfu operations
            cache.pop_lfu();
            assert_eq!(cache.len(), 1);

            cache.pop_lfu();
            assert_eq!(cache.len(), 0);

            // Test pop_lfu on empty cache doesn't change length
            assert_eq!(cache.pop_lfu(), None);
            assert_eq!(cache.len(), 0);

            // Test clear operation
            cache.insert("test1".to_string(), Arc::new(1));
            cache.insert("test2".to_string(), Arc::new(2));
            assert_eq!(cache.len(), 2);

            cache.clear();
            assert_eq!(cache.len(), 0);

            // Test that get operations don't affect length
            cache.insert("key1".to_string(), Arc::new(100));
            cache.insert("key2".to_string(), Arc::new(200));
            assert_eq!(cache.len(), 2);

            cache.get(&"key1".to_string());
            cache.get(&"key2".to_string());
            cache.get(&"nonexistent".to_string());
            assert_eq!(cache.len(), 2); // Should remain unchanged
        }

        #[test]
        fn test_capacity_consistency() {
            // Test different capacity values
            let capacities = [0, 1, 3, 10, 100];

            for &capacity in &capacities {
                let mut cache = LFUCache::<String, i32>::new(capacity);

                // Test initial capacity
                assert_eq!(cache.capacity(), capacity);

                // Test capacity doesn't change after operations
                if capacity > 0 {
                    // Insert items up to capacity
                    for i in 0..capacity {
                        cache.insert(format!("key{}", i), Arc::new(i as i32));
                        assert_eq!(cache.capacity(), capacity); // Should never change
                        assert!(cache.len() <= capacity); // Should never exceed capacity
                    }

                    // Insert beyond capacity
                    for i in capacity..(capacity + 5) {
                        cache.insert(format!("key{}", i), Arc::new(i as i32));
                        assert_eq!(cache.capacity(), capacity); // Should never change
                        assert_eq!(cache.len(), capacity); // Should stay at capacity
                    }

                    // Test other operations don't change capacity
                    cache.get(&format!("key{}", capacity - 1));
                    assert_eq!(cache.capacity(), capacity);

                    cache.remove(&format!("key{}", capacity - 1));
                    assert_eq!(cache.capacity(), capacity);

                    cache.pop_lfu();
                    assert_eq!(cache.capacity(), capacity);

                    cache.clear();
                    assert_eq!(cache.capacity(), capacity);
                } else {
                    // Test zero capacity case
                    assert_eq!(cache.capacity(), 0);
                    cache.insert("key1".to_string(), Arc::new(100));
                    assert_eq!(cache.len(), 0); // Should remain empty
                    assert_eq!(cache.capacity(), 0); // Should remain 0
                }
            }

            // Test capacity consistency across multiple operations
            let mut cache = LFUCache::new(5);
            let original_capacity = cache.capacity();

            // Perform 100 random operations
            for i in 0..100 {
                match i % 4 {
                    0 => {
                        cache.insert(format!("key{}", i % 10), Arc::new(i));
                    },
                    1 => {
                        cache.get(&format!("key{}", i % 10));
                    },
                    2 => {
                        cache.remove(&format!("key{}", i % 10));
                    },
                    3 => {
                        cache.pop_lfu();
                    },
                    _ => unreachable!(),
                }

                // Verify capacity never changes and constraints are respected
                assert_eq!(cache.capacity(), original_capacity);
                assert!(cache.len() <= cache.capacity());
            }
        }

        #[test]
        fn test_clear_resets_all_state() {
            let mut cache = LFUCache::new(5);

            // Populate cache with data and complex state
            cache.insert("key1".to_string(), Arc::new(100));
            cache.insert("key2".to_string(), Arc::new(200));
            cache.insert("key3".to_string(), Arc::new(300));
            cache.insert("key4".to_string(), Arc::new(400));
            cache.insert("key5".to_string(), Arc::new(500));

            // Create complex frequency patterns
            for _ in 0..10 {
                cache.get(&"key1".to_string());
            }
            for _ in 0..5 {
                cache.get(&"key2".to_string());
            }
            for _ in 0..3 {
                cache.get(&"key3".to_string());
            }
            cache.get(&"key4".to_string());
            // key5 remains at frequency 1

            // Verify complex state exists
            assert_eq!(cache.len(), 5);
            assert_eq!(cache.frequency(&"key1".to_string()), Some(11)); // 1 + 10
            assert_eq!(cache.frequency(&"key2".to_string()), Some(6)); // 1 + 5
            assert_eq!(cache.frequency(&"key3".to_string()), Some(4)); // 1 + 3
            assert_eq!(cache.frequency(&"key4".to_string()), Some(2)); // 1 + 1
            assert_eq!(cache.frequency(&"key5".to_string()), Some(1)); // 1 + 0

            // Clear the cache
            cache.clear();

            // Verify complete state reset
            assert_eq!(cache.len(), 0);
            assert_eq!(cache.capacity(), 5); // Capacity should remain unchanged

            // Verify all keys are gone
            assert!(!cache.contains(&"key1".to_string()));
            assert!(!cache.contains(&"key2".to_string()));
            assert!(!cache.contains(&"key3".to_string()));
            assert!(!cache.contains(&"key4".to_string()));
            assert!(!cache.contains(&"key5".to_string()));

            // Verify all frequencies are gone
            assert_eq!(cache.frequency(&"key1".to_string()), None);
            assert_eq!(cache.frequency(&"key2".to_string()), None);
            assert_eq!(cache.frequency(&"key3".to_string()), None);
            assert_eq!(cache.frequency(&"key4".to_string()), None);
            assert_eq!(cache.frequency(&"key5".to_string()), None);

            // Verify get operations return None
            assert_eq!(cache.get(&"key1".to_string()), None);
            assert_eq!(cache.get(&"key2".to_string()), None);

            // Verify LFU operations work on empty cache
            assert_eq!(cache.pop_lfu(), None);
            assert_eq!(cache.peek_lfu(), None);

            // Verify cache is ready for fresh use
            cache.insert("new_key".to_string(), Arc::new(999));
            assert_eq!(cache.len(), 1);
            assert_eq!(cache.frequency(&"new_key".to_string()), Some(1));
            assert_eq!(
                cache.get(&"new_key".to_string()).map(Arc::as_ref),
                Some(&999)
            );

            // Test multiple clears are safe
            cache.clear();
            assert_eq!(cache.len(), 0);

            cache.clear(); // Second clear on empty cache
            assert_eq!(cache.len(), 0);
            assert_eq!(cache.capacity(), 5); // Capacity still preserved

            // Test clear after partial population
            cache.insert("test1".to_string(), Arc::new(1));
            cache.insert("test2".to_string(), Arc::new(2));
            assert_eq!(cache.len(), 2);

            cache.clear();
            assert_eq!(cache.len(), 0);
            assert_eq!(cache.frequency(&"test1".to_string()), None);
            assert_eq!(cache.frequency(&"test2".to_string()), None);
        }

        #[test]
        fn test_remove_consistency() {
            let mut cache = LFUCache::new(5);

            // Setup cache with various frequencies
            cache.insert("key1".to_string(), Arc::new(100));
            cache.insert("key2".to_string(), Arc::new(200));
            cache.insert("key3".to_string(), Arc::new(300));
            cache.insert("key4".to_string(), Arc::new(400));

            // Create different frequency patterns
            cache.get(&"key1".to_string()); // key1: freq = 2
            cache.get(&"key1".to_string()); // key1: freq = 3
            cache.get(&"key2".to_string()); // key2: freq = 2
            cache.get(&"key3".to_string()); // key3: freq = 2
            // key4: freq = 1

            assert_eq!(cache.len(), 4);

            // Test successful removal
            let removed_value = cache.remove(&"key2".to_string());
            assert_eq!(removed_value.as_deref(), Some(&200));
            assert_eq!(cache.len(), 3);

            // Verify key is completely gone
            assert!(!cache.contains(&"key2".to_string()));
            assert_eq!(cache.get(&"key2".to_string()), None);
            assert_eq!(cache.frequency(&"key2".to_string()), None);

            // Verify other keys are unaffected
            assert!(cache.contains(&"key1".to_string()));
            assert!(cache.contains(&"key3".to_string()));
            assert!(cache.contains(&"key4".to_string()));
            assert_eq!(cache.frequency(&"key1".to_string()), Some(3));
            assert_eq!(cache.frequency(&"key3".to_string()), Some(2));
            assert_eq!(cache.frequency(&"key4".to_string()), Some(1));

            // Test removal of non-existent key
            let removed_none = cache.remove(&"nonexistent".to_string());
            assert_eq!(removed_none, None);
            assert_eq!(cache.len(), 3); // Should remain unchanged

            // Test removal of key with highest frequency
            let removed_high_freq = cache.remove(&"key1".to_string());
            assert_eq!(removed_high_freq.as_deref(), Some(&100));
            assert_eq!(cache.len(), 2);
            assert_eq!(cache.frequency(&"key1".to_string()), None);

            // Test removal of key with lowest frequency
            let removed_low_freq = cache.remove(&"key4".to_string());
            assert_eq!(removed_low_freq.as_deref(), Some(&400));
            assert_eq!(cache.len(), 1);
            assert_eq!(cache.frequency(&"key4".to_string()), None);

            // Verify LFU operations still work correctly after removals
            let (lfu_key, lfu_value) = cache.peek_lfu().unwrap();
            assert_eq!(lfu_key, &"key3".to_string());
            assert_eq!(lfu_value.as_ref(), &300);

            // Test removing the last item
            let removed_last = cache.remove(&"key3".to_string());
            assert_eq!(removed_last.as_deref(), Some(&300));
            assert_eq!(cache.len(), 0);

            // Verify empty cache state
            assert_eq!(cache.peek_lfu(), None);
            assert_eq!(cache.pop_lfu(), None);

            // Test removal on empty cache
            let removed_from_empty = cache.remove(&"key1".to_string());
            assert_eq!(removed_from_empty, None);
            assert_eq!(cache.len(), 0);

            // Test cache functionality after complete emptying via removals
            cache.insert("new_key".to_string(), Arc::new(999));
            assert_eq!(cache.len(), 1);
            assert_eq!(cache.frequency(&"new_key".to_string()), Some(1));

            // Test removing and re-inserting same key
            cache.remove(&"new_key".to_string());
            assert_eq!(cache.len(), 0);

            cache.insert("new_key".to_string(), Arc::new(888));
            assert_eq!(cache.len(), 1);
            assert_eq!(cache.frequency(&"new_key".to_string()), Some(1)); // Fresh frequency
            assert_eq!(
                cache.get(&"new_key".to_string()).map(Arc::as_ref),
                Some(&888)
            );
        }

        #[test]
        fn test_eviction_consistency() {
            let mut cache = LFUCache::new(3);

            // Fill cache to capacity
            cache.insert("key1".to_string(), Arc::new(100));
            cache.insert("key2".to_string(), Arc::new(200));
            cache.insert("key3".to_string(), Arc::new(300));
            assert_eq!(cache.len(), 3);

            // Create frequency differences
            cache.get(&"key1".to_string()); // key1: freq = 2
            cache.get(&"key1".to_string()); // key1: freq = 3
            cache.get(&"key2".to_string()); // key2: freq = 2
            // key3: freq = 1 (lowest)

            // Insert beyond capacity - should evict key3 (LFU)
            cache.insert("key4".to_string(), Arc::new(400));
            assert_eq!(cache.len(), 3); // Should remain at capacity

            // Verify eviction occurred correctly
            assert!(!cache.contains(&"key3".to_string()));
            assert_eq!(cache.frequency(&"key3".to_string()), None);

            // Verify remaining items are correct
            assert!(cache.contains(&"key1".to_string()));
            assert!(cache.contains(&"key2".to_string()));
            assert!(cache.contains(&"key4".to_string()));
            assert_eq!(cache.frequency(&"key1".to_string()), Some(3));
            assert_eq!(cache.frequency(&"key2".to_string()), Some(2));
            assert_eq!(cache.frequency(&"key4".to_string()), Some(1)); // New item

            // Test eviction with tie-breaking
            cache.insert("key5".to_string(), Arc::new(500));
            assert_eq!(cache.len(), 3);

            // Either key4 or key5 should be evicted (both have freq=1)
            // But one of them should remain
            let has_key4 = cache.contains(&"key4".to_string());
            let has_key5 = cache.contains(&"key5".to_string());
            assert!(has_key4 ^ has_key5); // Exactly one should be true (XOR)

            // High frequency items should always remain
            assert!(cache.contains(&"key1".to_string()));
            assert!(cache.contains(&"key2".to_string()));

            // Test multiple evictions
            cache.insert("key6".to_string(), Arc::new(600));
            cache.insert("key7".to_string(), Arc::new(700));
            assert_eq!(cache.len(), 3); // Should still be at capacity

            // key1 and key2 should still be there due to higher frequency
            assert!(cache.contains(&"key1".to_string()));
            assert!(cache.contains(&"key2".to_string()));

            // Test eviction doesn't break LFU ordering
            #[cfg(debug_assertions)]
            cache.debug_validate_invariants();

            // Test eviction with zero capacity
            let mut zero_cache = LFUCache::<String, i32>::new(0);
            zero_cache.insert("key1".to_string(), Arc::new(100));
            assert_eq!(zero_cache.len(), 0); // Should reject insertion
            assert!(!zero_cache.contains(&"key1".to_string()));

            // Test eviction preserves invariants
            let mut test_cache = LFUCache::new(2);

            // Insert items with known frequencies
            test_cache.insert("low".to_string(), Arc::new(1));
            test_cache.insert("high".to_string(), Arc::new(2));

            // Make high frequency item
            for _ in 0..5 {
                test_cache.get(&"high".to_string());
            }

            // Insert new item - should evict "low"
            test_cache.insert("new".to_string(), Arc::new(3));
            assert_eq!(test_cache.len(), 2);
            assert!(!test_cache.contains(&"low".to_string()));
            assert!(test_cache.contains(&"high".to_string()));
            assert!(test_cache.contains(&"new".to_string()));

            // Verify frequencies are consistent after eviction
            assert_eq!(test_cache.frequency(&"low".to_string()), None);
            assert!(test_cache.frequency(&"high".to_string()).unwrap() > 1);
            assert_eq!(test_cache.frequency(&"new".to_string()), Some(1));
        }

        #[test]
        fn test_frequency_increment_on_get() {
            let mut cache = LFUCache::new(5);

            // Insert items with initial frequency of 1
            cache.insert("key1".to_string(), Arc::new(100));
            cache.insert("key2".to_string(), Arc::new(200));
            cache.insert("key3".to_string(), Arc::new(300));

            // Verify initial frequencies
            assert_eq!(cache.frequency(&"key1".to_string()), Some(1));
            assert_eq!(cache.frequency(&"key2".to_string()), Some(1));
            assert_eq!(cache.frequency(&"key3".to_string()), Some(1));

            // Test single get operations
            assert_eq!(cache.get(&"key1".to_string()).map(Arc::as_ref), Some(&100));
            assert_eq!(cache.frequency(&"key1".to_string()), Some(2));

            assert_eq!(cache.get(&"key2".to_string()).map(Arc::as_ref), Some(&200));
            assert_eq!(cache.frequency(&"key2".to_string()), Some(2));

            // Test multiple get operations on same key
            assert_eq!(cache.get(&"key1".to_string()).map(Arc::as_ref), Some(&100));
            assert_eq!(cache.frequency(&"key1".to_string()), Some(3));

            assert_eq!(cache.get(&"key1".to_string()).map(Arc::as_ref), Some(&100));
            assert_eq!(cache.frequency(&"key1".to_string()), Some(4));

            // Test get on non-existent key doesn't create entry
            assert_eq!(cache.get(&"nonexistent".to_string()), None);
            assert_eq!(cache.frequency(&"nonexistent".to_string()), None);
            assert_eq!(cache.len(), 3); // Should remain unchanged

            // Test frequency increments are independent per key
            for _ in 0..10 {
                cache.get(&"key2".to_string());
            }
            for _ in 0..5 {
                cache.get(&"key3".to_string());
            }

            assert_eq!(cache.frequency(&"key1".to_string()), Some(4)); // Unchanged
            assert_eq!(cache.frequency(&"key2".to_string()), Some(12)); // 2 + 10
            assert_eq!(cache.frequency(&"key3".to_string()), Some(6)); // 1 + 5

            // Test get after insert update preserves frequency
            cache.insert("key1".to_string(), Arc::new(999)); // Update value
            assert_eq!(cache.frequency(&"key1".to_string()), Some(4)); // Frequency preserved
            assert_eq!(cache.get(&"key1".to_string()).map(Arc::as_ref), Some(&999)); // New value
            assert_eq!(cache.frequency(&"key1".to_string()), Some(5)); // Frequency incremented

            // Test frequency increments affect LFU ordering
            cache.insert("key4".to_string(), Arc::new(400));
            assert_eq!(cache.frequency(&"key4".to_string()), Some(1)); // New item

            // key4 should be LFU now
            let (lfu_key, _) = cache.peek_lfu().unwrap();
            assert_eq!(lfu_key, &"key4".to_string());

            // After accessing key4, it should no longer be LFU
            cache.get(&"key4".to_string());
            cache.get(&"key4".to_string());
            assert_eq!(cache.frequency(&"key4".to_string()), Some(3));

            // Insert a new item that will become the new LFU
            cache.insert("key5".to_string(), Arc::new(500));
            assert_eq!(cache.frequency(&"key5".to_string()), Some(1));

            // Now key5 should be LFU (frequency = 1)
            let (new_lfu_key, _) = cache.peek_lfu().unwrap();
            assert_eq!(new_lfu_key, &"key5".to_string());
            let new_lfu_freq = cache.frequency(new_lfu_key).unwrap();
            assert_eq!(new_lfu_freq, 1);

            // Test rapid frequency increments
            let initial_freq = cache.frequency(&"key1".to_string()).unwrap();
            for i in 1..=100 {
                cache.get(&"key1".to_string());
                assert_eq!(cache.frequency(&"key1".to_string()), Some(initial_freq + i));
            }

            // Test that get operations don't affect other keys' frequencies
            let key2_freq_before = cache.frequency(&"key2".to_string()).unwrap();
            let key3_freq_before = cache.frequency(&"key3".to_string()).unwrap();
            let key4_freq_before = cache.frequency(&"key4".to_string()).unwrap();

            cache.get(&"key1".to_string()); // Only affect key1

            assert_eq!(cache.frequency(&"key2".to_string()), Some(key2_freq_before));
            assert_eq!(cache.frequency(&"key3".to_string()), Some(key3_freq_before));
            assert_eq!(cache.frequency(&"key4".to_string()), Some(key4_freq_before));
        }

        #[test]
        fn test_invariants_after_operations() {
            let mut cache = LFUCache::new(4);

            // Helper function to verify all invariants
            let verify_invariants = |cache: &mut LFUCache<String, i32>| {
                if cache.len() > 0 {
                    assert!(cache.peek_lfu().is_some());
                } else {
                    assert!(cache.peek_lfu().is_none());
                }

                let test_keys = ["key1", "key2", "key3", "key4", "key5", "nonexistent"];
                for key in test_keys {
                    let contains_result = cache.contains(&key.to_string());
                    let get_result = cache.get(&key.to_string()).is_some();
                    assert_eq!(contains_result, get_result);
                }

                #[cfg(debug_assertions)]
                cache.debug_validate_invariants();
            };

            // Test invariants after initial state
            verify_invariants(&mut cache);

            // Test invariants after insertions
            cache.insert("key1".to_string(), Arc::new(100));
            verify_invariants(&mut cache);

            cache.insert("key2".to_string(), Arc::new(200));
            verify_invariants(&mut cache);

            cache.insert("key3".to_string(), Arc::new(300));
            verify_invariants(&mut cache);

            cache.insert("key4".to_string(), Arc::new(400));
            verify_invariants(&mut cache);

            // Test invariants after gets (frequency changes)
            cache.get(&"key1".to_string());
            verify_invariants(&mut cache);

            cache.get(&"key1".to_string());
            cache.get(&"key2".to_string());
            verify_invariants(&mut cache);

            // Test invariants after capacity overflow (eviction)
            cache.insert("key5".to_string(), Arc::new(500));
            verify_invariants(&mut cache);

            // Test invariants after multiple operations
            for i in 0..20 {
                match i % 5 {
                    0 => {
                        cache.insert(format!("temp{}", i), Arc::new(i));
                    },
                    1 => {
                        cache.get(&"key1".to_string());
                    },
                    2 => {
                        cache.remove(&format!("temp{}", i - 1));
                    },
                    3 => {
                        cache.pop_lfu();
                    },
                    4 => {
                        cache.increment_frequency(&"key2".to_string());
                    },
                    _ => unreachable!(),
                }
                verify_invariants(&mut cache);
            }

            // Test invariants after frequency manipulations
            cache.reset_frequency(&"key1".to_string());
            verify_invariants(&mut cache);

            cache.increment_frequency(&"key2".to_string());
            verify_invariants(&mut cache);

            // Test invariants after removals
            for candidate in ["key1", "key2", "key3", "key4"] {
                if cache.contains(&candidate.to_string()) {
                    cache.remove(&candidate.to_string());
                    verify_invariants(&mut cache);
                }
            }

            // Test invariants after pop_lfu operations
            while cache.len() > 0 {
                cache.pop_lfu();
                verify_invariants(&mut cache);
            }

            // Test invariants after clear
            cache.insert("test1".to_string(), Arc::new(1));
            cache.insert("test2".to_string(), Arc::new(2));
            verify_invariants(&mut cache);

            cache.clear();
            verify_invariants(&mut cache);

            // Test invariants with edge cases

            // Zero capacity cache
            let mut zero_cache = LFUCache::<String, i32>::new(0);
            verify_invariants(&mut zero_cache);
            zero_cache.insert("test".to_string(), Arc::new(1));
            verify_invariants(&mut zero_cache);

            // Single capacity cache
            let mut single_cache = LFUCache::new(1);
            verify_invariants(&mut single_cache);

            single_cache.insert("only".to_string(), Arc::new(1));
            verify_invariants(&mut single_cache);

            single_cache.insert("replace".to_string(), Arc::new(2));
            verify_invariants(&mut single_cache);

            // Test with complex frequency patterns
            let mut complex_cache = LFUCache::new(3);
            complex_cache.insert("a".to_string(), Arc::new(1));
            complex_cache.insert("b".to_string(), Arc::new(2));
            complex_cache.insert("c".to_string(), Arc::new(3));

            // Create Fibonacci-like frequency pattern
            for i in 1..=10 {
                for _ in 0..i {
                    complex_cache.get(&"a".to_string());
                }
                for _ in 0..(i / 2) {
                    complex_cache.get(&"b".to_string());
                }
                verify_invariants(&mut complex_cache);
            }
        }
    }
}
