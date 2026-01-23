//! Frequency buckets for O(1) LFU tracking.
//!
//! Provides LFU (Least Frequently Used) eviction metadata tracking with O(1)
//! insert, touch, remove, and eviction operations. Uses frequency buckets with
//! FIFO tie-breaking within each frequency level.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                        FrequencyBuckets<K> Layout                           │
//! │                                                                             │
//! │   ┌─────────────────────────────┐   ┌─────────────────────────────────┐   │
//! │   │  index: HashMap<K, SlotId>  │   │  entries: SlotArena<Entry<K>>   │   │
//! │   │                             │   │                                 │   │
//! │   │  ┌───────────┬──────────┐  │   │  ┌──────┬───────────────────┐  │   │
//! │   │  │    Key    │  SlotId  │  │   │  │ Slot │ Entry             │  │   │
//! │   │  ├───────────┼──────────┤  │   │  ├──────┼───────────────────┤  │   │
//! │   │  │  "page_a" │   id_0   │──┼───┼──►│ id_0 │ freq:2, prev/next │  │   │
//! │   │  │  "page_b" │   id_1   │──┼───┼──►│ id_1 │ freq:1, prev/next │  │   │
//! │   │  │  "page_c" │   id_2   │──┼───┼──►│ id_2 │ freq:1, prev/next │  │   │
//! │   │  └───────────┴──────────┘  │   │  └──────┴───────────────────┘  │   │
//! │   └─────────────────────────────┘   └─────────────────────────────────┘   │
//! │                                                                             │
//! │   ┌───────────────────────────────────────────────────────────────────┐   │
//! │   │  buckets: HashMap<u64, Bucket>  (frequency → doubly-linked list)  │   │
//! │   │                                                                   │   │
//! │   │  min_freq = 1                                                     │   │
//! │   │       │                                                           │   │
//! │   │       ▼                                                           │   │
//! │   │  freq=1: head ──► [id_2] ◄──► [id_1] ◄── tail  (FIFO order)      │   │
//! │   │                     MRU          LRU (evict first)                │   │
//! │   │                                                                   │   │
//! │   │  freq=2: head ──► [id_0] ◄── tail                                │   │
//! │   │                                                                   │   │
//! │   │  Bucket links: freq=1 ──next──► freq=2                           │   │
//! │   │                freq=2 ◄──prev── freq=1                           │   │
//! │   └───────────────────────────────────────────────────────────────────┘   │
//! │                                                                             │
//! └─────────────────────────────────────────────────────────────────────────────┘
//!
//! Touch Flow (increment frequency)
//! ─────────────────────────────────
//!
//!   touch("page_b"):
//!     1. Lookup id_1 in index
//!     2. Remove id_1 from freq=1 bucket list
//!     3. If freq=1 bucket empty → remove bucket, update min_freq
//!     4. Create freq=2 bucket if needed
//!     5. Push id_1 to front of freq=2 bucket (MRU)
//!     6. Update entry.freq = 2
//!
//! Eviction Flow (pop_min)
//! ───────────────────────
//!
//!   pop_min():
//!     1. Use min_freq to find lowest bucket
//!     2. Pop tail of that bucket (oldest at that frequency)
//!     3. Remove entry from index and entries
//!     4. If bucket empty → remove bucket, update min_freq
//!     5. Return (key, freq)
//! ```
//!
//! ## Key Components
//!
//! - [`FrequencyBuckets`]: Single-threaded O(1) LFU tracker
//! - [`ShardedFrequencyBuckets`]: Concurrent sharded variant
//! - [`FrequencyBucketsHandle`]: Handle-based variant for interned keys
//!
//! ## Operations
//!
//! | Operation      | Time        | Notes                                  |
//! |----------------|-------------|----------------------------------------|
//! | `insert`       | O(1)        | New key starts at freq=1               |
//! | `touch`        | O(1)        | Increment frequency, move to MRU       |
//! | `remove`       | O(1)        | Remove from tracking                   |
//! | `pop_min`      | O(1)        | Evict LFU (FIFO tie-break)             |
//! | `frequency`    | O(1)        | Query current frequency                |
//! | `decay_halve`  | O(n)        | Halve all frequencies                  |
//! | `rebase_min_freq` | O(n)     | Rebase so min becomes 1                |
//!
//! ## Use Cases
//!
//! - **LFU cache policy**: Track access frequency for eviction
//! - **Hot/cold detection**: Identify frequently accessed items
//! - **Admission control**: Filter one-hit wonders
//!
//! ## Example Usage
//!
//! ```
//! use cachekit::ds::FrequencyBuckets;
//!
//! let mut freq = FrequencyBuckets::new();
//!
//! // Insert keys (all start at frequency 1)
//! freq.insert("page_a");
//! freq.insert("page_b");
//! freq.insert("page_c");
//!
//! // Access increases frequency
//! freq.touch(&"page_a");  // freq=2
//! freq.touch(&"page_a");  // freq=3
//!
//! // Evict LFU (lowest frequency, FIFO among ties)
//! let evicted = freq.pop_min();
//! assert_eq!(evicted, Some(("page_b", 1)));  // First inserted at freq=1
//! ```
//!
//! ## Use Case: LFU Cache with Frequency Decay
//!
//! ```
//! use cachekit::ds::FrequencyBuckets;
//!
//! struct LfuCache {
//!     freq: FrequencyBuckets<String>,
//!     decay_interval: u64,
//!     ops_since_decay: u64,
//! }
//!
//! impl LfuCache {
//!     fn new(decay_interval: u64) -> Self {
//!         Self {
//!             freq: FrequencyBuckets::new(),
//!             decay_interval,
//!             ops_since_decay: 0,
//!         }
//!     }
//!
//!     fn access(&mut self, key: &str) {
//!         if !self.freq.contains(&key.to_string()) {
//!             self.freq.insert(key.to_string());
//!         } else {
//!             self.freq.touch(&key.to_string());
//!         }
//!
//!         self.ops_since_decay += 1;
//!         if self.ops_since_decay >= self.decay_interval {
//!             self.freq.decay_halve();  // Prevent frequency inflation
//!             self.ops_since_decay = 0;
//!         }
//!     }
//!
//!     fn evict(&mut self) -> Option<String> {
//!         self.freq.pop_min().map(|(k, _)| k)
//!     }
//! }
//!
//! let mut cache = LfuCache::new(100);
//! cache.access("hot_page");
//! cache.access("hot_page");
//! cache.access("cold_page");
//!
//! assert_eq!(cache.freq.frequency(&"hot_page".to_string()), Some(2));
//! ```
//!
//! ## Handle-Based Usage
//!
//! For large keys, consider interning keys in a higher layer and using
//! [`FrequencyBucketsHandle<Handle>`] where `Handle: Copy + Eq + Hash`.
//! This stores the handle (not the full key) in buckets and index maps.
//!
//! ## Thread Safety
//!
//! - [`FrequencyBuckets`]: Not thread-safe
//! - [`ShardedFrequencyBuckets`]: Thread-safe via sharding with `RwLock`
//!
//! ## Implementation Notes
//!
//! - Buckets are doubly-linked for O(1) navigation
//! - FIFO within bucket: head=MRU, tail=LRU (evict from tail)
//! - `min_freq` pointer enables O(1) eviction
//! - `debug_validate_invariants()` available in debug/test builds

use rustc_hash::FxHashMap;
use std::hash::Hash;

use crate::ds::slot_arena::{SlotArena, SlotId};

/// LFU entry with cache-line optimized layout.
/// Link pointers (prev/next) are accessed on every touch/evict operation,
/// so they're placed first for better cache locality.
#[derive(Debug)]
#[repr(C)]
struct Entry<K> {
    // Hot fields - accessed during list operations
    prev: Option<SlotId>,
    next: Option<SlotId>,
    freq: u64,
    last_epoch: u64,
    // Cold field - only accessed on eviction
    key: K,
}

#[derive(Debug, Default)]
struct Bucket {
    head: Option<SlotId>,
    tail: Option<SlotId>,
    prev: Option<u64>,
    next: Option<u64>,
}

/// O(1) LFU metadata tracker with FIFO tie-breaking within a frequency.
///
/// Tracks key frequencies for LFU eviction. Keys are organized into frequency
/// buckets, with FIFO ordering within each bucket for tie-breaking.
///
/// # Type Parameters
///
/// - `K`: Key type, must be `Eq + Hash + Clone`
///
/// # Example
///
/// ```
/// use cachekit::ds::FrequencyBuckets;
///
/// let mut freq = FrequencyBuckets::new();
///
/// // Insert and touch
/// freq.insert("a");
/// freq.insert("b");
/// freq.touch(&"a");  // "a" now at freq=2
///
/// // Query state
/// assert_eq!(freq.frequency(&"a"), Some(2));
/// assert_eq!(freq.frequency(&"b"), Some(1));
/// assert_eq!(freq.min_freq(), Some(1));
///
/// // Evict LFU
/// let (key, freq_val) = freq.pop_min().unwrap();
/// assert_eq!(key, "b");
/// assert_eq!(freq_val, 1);
/// ```
///
/// # Use Case: Admission Filter
///
/// ```
/// use cachekit::ds::FrequencyBuckets;
///
/// // Only admit keys that have been seen multiple times
/// let mut freq: FrequencyBuckets<String> = FrequencyBuckets::new();
///
/// fn should_admit(freq: &mut FrequencyBuckets<String>, key: &str) -> bool {
///     let key_str = key.to_string();
///     if freq.contains(&key_str) {
///         let new_freq = freq.touch(&key_str).unwrap();
///         new_freq >= 2  // Admit after 2nd access
///     } else {
///         freq.insert(key_str);
///         false  // Don't admit on first access
///     }
/// }
///
/// assert!(!should_admit(&mut freq, "page_x"));  // First access
/// assert!(should_admit(&mut freq, "page_x"));   // Second access - admit!
/// ```
#[derive(Debug)]
pub struct FrequencyBuckets<K> {
    entries: SlotArena<Entry<K>>,
    index: FxHashMap<K, SlotId>,
    buckets: FxHashMap<u64, Bucket>,
    min_freq: u64,
    epoch: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Read-only view of a frequency bucket entry.
pub struct FrequencyBucketEntryMeta<'a, K> {
    pub key: &'a K,
    pub freq: u64,
    pub last_epoch: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Owned view of a frequency bucket entry for sharded readers.
pub struct ShardedFrequencyBucketEntryMeta<K> {
    pub key: K,
    pub freq: u64,
    pub last_epoch: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Debug view of a bucket entry for snapshots.
pub struct FrequencyBucketEntryDebug<K> {
    pub id: SlotId,
    pub key: K,
    pub freq: u64,
    pub last_epoch: u64,
}

/// Default bucket pre-allocation for typical frequency distributions.
/// Most items cluster at low frequencies (1-32), so 32 buckets covers most cases.
pub const DEFAULT_BUCKET_PREALLOC: usize = 32;

impl<K> FrequencyBuckets<K>
where
    K: Eq + Hash + Clone,
{
    /// Creates an empty tracker with reserved capacity for entries and index.
    ///
    /// Uses [`DEFAULT_BUCKET_PREALLOC`] for the bucket map. For custom bucket
    /// pre-allocation, use [`with_capacity_and_bucket_hint`](Self::with_capacity_and_bucket_hint).
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBuckets;
    ///
    /// let freq: FrequencyBuckets<String> = FrequencyBuckets::with_capacity(1000);
    /// assert!(freq.is_empty());
    /// ```
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_bucket_hint(capacity, DEFAULT_BUCKET_PREALLOC)
    }

    /// Creates an empty tracker with reserved capacity and custom bucket pre-allocation.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Pre-allocated space for entries and index
    /// * `bucket_hint` - Pre-allocated space for frequency buckets (number of distinct frequencies)
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBuckets;
    ///
    /// // Expect many distinct frequencies (e.g., long-running cache with varied access patterns)
    /// let freq: FrequencyBuckets<String> = FrequencyBuckets::with_capacity_and_bucket_hint(1000, 64);
    /// assert!(freq.is_empty());
    /// ```
    pub fn with_capacity_and_bucket_hint(capacity: usize, bucket_hint: usize) -> Self {
        Self {
            entries: SlotArena::with_capacity(capacity),
            index: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
            buckets: FxHashMap::with_capacity_and_hasher(bucket_hint, Default::default()),
            min_freq: 0,
            epoch: 0,
        }
    }

    /// Creates an empty tracker.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBuckets;
    ///
    /// let freq: FrequencyBuckets<&str> = FrequencyBuckets::new();
    /// assert!(freq.is_empty());
    /// ```
    pub fn new() -> Self {
        Self {
            entries: SlotArena::new(),
            index: FxHashMap::default(),
            buckets: FxHashMap::default(),
            min_freq: 0,
            epoch: 0,
        }
    }

    /// Returns the number of tracked keys.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBuckets;
    ///
    /// let mut freq = FrequencyBuckets::new();
    /// assert_eq!(freq.len(), 0);
    ///
    /// freq.insert("a");
    /// freq.insert("b");
    /// assert_eq!(freq.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if there are no tracked keys.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBuckets;
    ///
    /// let mut freq: FrequencyBuckets<&str> = FrequencyBuckets::new();
    /// assert!(freq.is_empty());
    ///
    /// freq.insert("key");
    /// assert!(!freq.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns `true` if `key` is present.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBuckets;
    ///
    /// let mut freq = FrequencyBuckets::new();
    /// freq.insert("key");
    ///
    /// assert!(freq.contains(&"key"));
    /// assert!(!freq.contains(&"missing"));
    /// ```
    #[inline]
    pub fn contains(&self, key: &K) -> bool {
        self.index.contains_key(key)
    }

    /// Returns the current epoch.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBuckets;
    ///
    /// let freq: FrequencyBuckets<&str> = FrequencyBuckets::new();
    /// assert_eq!(freq.current_epoch(), 0);
    /// ```
    pub fn current_epoch(&self) -> u64 {
        self.epoch
    }

    /// Advances the epoch counter and returns the new value.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBuckets;
    ///
    /// let mut freq: FrequencyBuckets<&str> = FrequencyBuckets::new();
    /// assert_eq!(freq.advance_epoch(), 1);
    /// assert_eq!(freq.advance_epoch(), 2);
    /// assert_eq!(freq.current_epoch(), 2);
    /// ```
    pub fn advance_epoch(&mut self) -> u64 {
        self.epoch = self.epoch.wrapping_add(1);
        self.epoch
    }

    /// Sets the epoch counter.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBuckets;
    ///
    /// let mut freq: FrequencyBuckets<&str> = FrequencyBuckets::new();
    /// freq.set_epoch(100);
    /// assert_eq!(freq.current_epoch(), 100);
    /// ```
    pub fn set_epoch(&mut self, epoch: u64) {
        self.epoch = epoch;
    }

    /// Returns the current frequency for `key`, if present.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBuckets;
    ///
    /// let mut freq = FrequencyBuckets::new();
    /// freq.insert("key");
    /// freq.touch(&"key");
    ///
    /// assert_eq!(freq.frequency(&"key"), Some(2));
    /// assert_eq!(freq.frequency(&"missing"), None);
    /// ```
    #[inline]
    pub fn frequency(&self, key: &K) -> Option<u64> {
        let id = *self.index.get(key)?;
        self.entries.get(id).map(|entry| entry.freq)
    }

    /// Returns the last epoch recorded for `key`.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBuckets;
    ///
    /// let mut freq = FrequencyBuckets::new();
    /// freq.set_epoch(10);
    /// freq.insert("key");
    ///
    /// assert_eq!(freq.entry_epoch(&"key"), Some(10));
    /// assert_eq!(freq.entry_epoch(&"missing"), None);
    /// ```
    pub fn entry_epoch(&self, key: &K) -> Option<u64> {
        let id = *self.index.get(key)?;
        self.entries.get(id).map(|entry| entry.last_epoch)
    }

    /// Sets the last epoch for `key`; returns `false` if missing.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBuckets;
    ///
    /// let mut freq = FrequencyBuckets::new();
    /// freq.insert("key");
    ///
    /// assert!(freq.set_entry_epoch(&"key", 42));
    /// assert_eq!(freq.entry_epoch(&"key"), Some(42));
    ///
    /// assert!(!freq.set_entry_epoch(&"missing", 42));
    /// ```
    pub fn set_entry_epoch(&mut self, key: &K, epoch: u64) -> bool {
        let id = match self.index.get(key) {
            Some(id) => *id,
            None => return false,
        };
        if let Some(entry) = self.entries.get_mut(id) {
            entry.last_epoch = epoch;
            return true;
        }
        false
    }

    /// Returns `true` if a borrowed key is present (avoids cloning).
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBuckets;
    ///
    /// let mut freq = FrequencyBuckets::new();
    /// freq.insert("hello".to_string());
    ///
    /// // Query with &str instead of String
    /// assert!(freq.contains_borrowed("hello"));
    /// ```
    pub fn contains_borrowed<Q>(&self, key: &Q) -> bool
    where
        K: std::borrow::Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        self.index.contains_key(key)
    }

    /// Returns the frequency for a borrowed key if present (avoids cloning).
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBuckets;
    ///
    /// let mut freq = FrequencyBuckets::new();
    /// freq.insert("hello".to_string());
    /// freq.touch(&"hello".to_string());
    ///
    /// // Query with &str instead of String
    /// assert_eq!(freq.frequency_borrowed("hello"), Some(2));
    /// assert_eq!(freq.frequency_borrowed("missing"), None);
    /// ```
    pub fn frequency_borrowed<Q>(&self, key: &Q) -> Option<u64>
    where
        K: std::borrow::Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        let id = *self.index.get(key)?;
        self.entries.get(id).map(|entry| entry.freq)
    }

    /// Returns the minimum frequency currently present.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBuckets;
    ///
    /// let mut freq = FrequencyBuckets::new();
    /// assert_eq!(freq.min_freq(), None);
    ///
    /// freq.insert("a");
    /// freq.insert("b");
    /// freq.touch(&"a");  // "a" at freq=2, "b" at freq=1
    ///
    /// assert_eq!(freq.min_freq(), Some(1));
    /// ```
    pub fn min_freq(&self) -> Option<u64> {
        if self.min_freq == 0 {
            None
        } else {
            Some(self.min_freq)
        }
    }

    /// Peeks the eviction candidate `(key, freq)` (tail of the min-frequency bucket).
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBuckets;
    ///
    /// let mut freq = FrequencyBuckets::new();
    /// freq.insert("a");
    /// freq.insert("b");
    /// freq.touch(&"b");  // "b" at freq=2
    ///
    /// // "a" is the eviction candidate (freq=1, oldest)
    /// let (key, freq_val) = freq.peek_min().unwrap();
    /// assert_eq!(*key, "a");
    /// assert_eq!(freq_val, 1);
    /// assert_eq!(freq.len(), 2);  // Not removed
    /// ```
    pub fn peek_min(&self) -> Option<(&K, u64)> {
        if self.min_freq == 0 {
            return None;
        }
        let min_freq = self.min_freq;
        let bucket = self.buckets.get(&min_freq)?;
        let id = bucket.tail?;
        let entry = self.entries.get(id)?;
        Some((&entry.key, entry.freq))
    }

    /// Peeks the SlotId for the eviction candidate (tail of the min-frequency bucket).
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBuckets;
    ///
    /// let mut freq = FrequencyBuckets::new();
    /// assert!(freq.peek_min_id().is_none());
    ///
    /// freq.insert("a");
    /// freq.insert("b");
    ///
    /// let id = freq.peek_min_id().unwrap();
    /// // The SlotId can be used to look up entry metadata
    /// ```
    pub fn peek_min_id(&self) -> Option<SlotId> {
        if self.min_freq == 0 {
            return None;
        }
        let bucket = self.buckets.get(&self.min_freq)?;
        bucket.tail
    }

    /// Peeks the key for the eviction candidate (tail of the min-frequency bucket).
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBuckets;
    ///
    /// let mut freq = FrequencyBuckets::new();
    /// freq.insert("a");
    /// freq.insert("b");
    /// freq.touch(&"b");
    ///
    /// // "a" is the eviction candidate (lowest freq, oldest)
    /// assert_eq!(freq.peek_min_key(), Some(&"a"));
    /// ```
    pub fn peek_min_key(&self) -> Option<&K> {
        let id = self.peek_min_id()?;
        self.entries.get(id).map(|entry| &entry.key)
    }

    /// Returns an iterator of SlotIds for a given frequency, from head to tail.
    ///
    /// Head is the most recently touched entry at that frequency (MRU),
    /// tail is the oldest (LRU, evict first).
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBuckets;
    ///
    /// let mut freq = FrequencyBuckets::new();
    /// freq.insert("a");
    /// freq.insert("b");
    /// freq.insert("c");
    ///
    /// let ids: Vec<_> = freq.iter_bucket_ids(1).collect();
    /// assert_eq!(ids.len(), 3);
    /// ```
    pub fn iter_bucket_ids(&self, freq: u64) -> FrequencyBucketIdIter<'_, K> {
        let head = self.buckets.get(&freq).and_then(|bucket| bucket.head);
        FrequencyBucketIdIter {
            buckets: self,
            current: head,
        }
    }

    /// Returns an iterator of `(SlotId, meta)` for a given frequency.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBuckets;
    ///
    /// let mut freq = FrequencyBuckets::new();
    /// freq.insert("a");
    /// freq.insert("b");
    /// freq.touch(&"a");  // "a" moves to freq=2
    ///
    /// // Only "b" is at frequency 1
    /// let entries: Vec<_> = freq.iter_bucket_entries(1).collect();
    /// assert_eq!(entries.len(), 1);
    /// assert_eq!(*entries[0].1.key, "b");
    /// ```
    pub fn iter_bucket_entries(&self, freq: u64) -> FrequencyBucketEntryIter<'_, K> {
        let head = self.buckets.get(&freq).and_then(|bucket| bucket.head);
        FrequencyBucketEntryIter {
            buckets: self,
            current: head,
        }
    }

    /// Returns an iterator over all `(SlotId, meta)` entries.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBuckets;
    ///
    /// let mut freq = FrequencyBuckets::new();
    /// freq.insert("a");
    /// freq.insert("b");
    /// freq.touch(&"a");
    ///
    /// let entries: Vec<_> = freq.iter_entries().collect();
    /// assert_eq!(entries.len(), 2);
    ///
    /// // Check we have both keys
    /// let keys: Vec<_> = entries.iter().map(|(_, m)| *m.key).collect();
    /// assert!(keys.contains(&"a"));
    /// assert!(keys.contains(&"b"));
    /// ```
    pub fn iter_entries(&self) -> impl Iterator<Item = (SlotId, FrequencyBucketEntryMeta<'_, K>)> {
        self.entries.iter().map(|(id, entry)| {
            (
                id,
                FrequencyBucketEntryMeta {
                    key: &entry.key,
                    freq: entry.freq,
                    last_epoch: entry.last_epoch,
                },
            )
        })
    }

    /// Inserts a new key with frequency 1.
    ///
    /// Returns `false` if the key already exists (no update performed).
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBuckets;
    ///
    /// let mut freq = FrequencyBuckets::new();
    ///
    /// assert!(freq.insert("a"));   // New key
    /// assert!(!freq.insert("a"));  // Already exists
    /// assert_eq!(freq.frequency(&"a"), Some(1));
    /// ```
    #[inline]
    pub fn insert(&mut self, key: K) -> bool {
        if self.index.contains_key(&key) {
            return false;
        }

        let id = self.entries.insert(Entry {
            key: key.clone(),
            freq: 1,
            last_epoch: self.epoch,
            prev: None,
            next: None,
        });
        self.index.insert(key, id);

        if !self.buckets.contains_key(&1) {
            let next = if self.min_freq == 0 {
                None
            } else {
                Some(self.min_freq)
            };
            self.insert_bucket(1, None, next);
        }

        self.list_push_front(1, id);
        if self.min_freq == 0 || self.min_freq > 1 {
            self.min_freq = 1;
        }
        true
    }

    /// Inserts a batch of keys; returns number of newly inserted keys.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBuckets;
    ///
    /// let mut freq = FrequencyBuckets::new();
    /// let inserted = freq.insert_batch(["a", "b", "c", "a"]);  // "a" duplicated
    ///
    /// assert_eq!(inserted, 3);  // Only 3 unique keys inserted
    /// assert_eq!(freq.len(), 3);
    /// ```
    pub fn insert_batch<I>(&mut self, keys: I) -> usize
    where
        I: IntoIterator<Item = K>,
    {
        let mut inserted = 0;
        for key in keys {
            if self.insert(key) {
                inserted += 1;
            }
        }
        inserted
    }

    /// Increments frequency for `key` and returns the new frequency.
    ///
    /// Returns `None` if `key` is missing. Within each frequency bucket,
    /// the key is treated as MRU by being pushed to the front.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBuckets;
    ///
    /// let mut freq = FrequencyBuckets::new();
    /// freq.insert("key");
    ///
    /// assert_eq!(freq.touch(&"key"), Some(2));
    /// assert_eq!(freq.touch(&"key"), Some(3));
    /// assert_eq!(freq.touch(&"missing"), None);
    /// ```
    #[inline]
    pub fn touch(&mut self, key: &K) -> Option<u64> {
        let id = *self.index.get(key)?;
        let current_freq = self.entries.get(id)?.freq;
        if current_freq == u64::MAX {
            self.list_remove(current_freq, id)?;
            self.list_push_front(current_freq, id);
            if let Some(entry) = self.entries.get_mut(id) {
                entry.last_epoch = self.epoch;
            }
            return Some(current_freq);
        }
        let next_freq = current_freq + 1;

        let (prev_freq, next_existing) = {
            let bucket = self.buckets.get(&current_freq)?;
            (bucket.prev, bucket.next)
        };

        self.list_remove(current_freq, id)?;
        let bucket_empty = self.bucket_is_empty(current_freq);

        if bucket_empty {
            self.remove_bucket(current_freq, prev_freq, next_existing);
            if self.min_freq == current_freq {
                self.min_freq = next_existing.unwrap_or(0);
            }
        }

        if !self.buckets.contains_key(&next_freq) {
            let prev = if bucket_empty {
                prev_freq
            } else {
                Some(current_freq)
            };
            let next = next_existing;
            self.insert_bucket(next_freq, prev, next);
        }

        if let Some(entry) = self.entries.get_mut(id) {
            entry.freq = next_freq;
            entry.last_epoch = self.epoch;
        }
        self.list_push_front(next_freq, id);
        if self.min_freq == 0 || next_freq < self.min_freq {
            self.min_freq = next_freq;
        }

        Some(next_freq)
    }

    /// Touches a batch of keys; returns number of keys found.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBuckets;
    ///
    /// let mut freq = FrequencyBuckets::new();
    /// freq.insert_batch(["a", "b", "c"]);
    ///
    /// let touched = freq.touch_batch(["a", "b", "missing"]);
    /// assert_eq!(touched, 2);  // Only "a" and "b" found
    /// ```
    pub fn touch_batch<I>(&mut self, keys: I) -> usize
    where
        I: IntoIterator<Item = K>,
    {
        let mut touched = 0;
        for key in keys {
            if self.touch(&key).is_some() {
                touched += 1;
            }
        }
        touched
    }

    /// Increments frequency for `key`, clamping at `max_freq`.
    ///
    /// If the key is already at `max_freq`, it is moved to the front of its
    /// bucket (MRU position) and the frequency is unchanged.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBuckets;
    ///
    /// let mut freq = FrequencyBuckets::new();
    /// freq.insert("key");
    ///
    /// assert_eq!(freq.touch_capped(&"key", 3), Some(2));
    /// assert_eq!(freq.touch_capped(&"key", 3), Some(3));
    /// assert_eq!(freq.touch_capped(&"key", 3), Some(3));  // Capped
    /// assert_eq!(freq.frequency(&"key"), Some(3));
    /// ```
    pub fn touch_capped(&mut self, key: &K, max_freq: u64) -> Option<u64> {
        let max_freq = max_freq.max(1);
        let id = *self.index.get(key)?;
        let current_freq = self.entries.get(id)?.freq;
        if current_freq >= max_freq {
            self.list_remove(current_freq, id)?;
            self.list_push_front(current_freq, id);
            if let Some(entry) = self.entries.get_mut(id) {
                entry.last_epoch = self.epoch;
            }
            return Some(current_freq);
        }

        let next_freq = current_freq + 1;
        let (prev_freq, next_existing) = {
            let bucket = self.buckets.get(&current_freq)?;
            (bucket.prev, bucket.next)
        };

        self.list_remove(current_freq, id)?;
        let bucket_empty = self.bucket_is_empty(current_freq);

        if bucket_empty {
            self.remove_bucket(current_freq, prev_freq, next_existing);
            if self.min_freq == current_freq {
                self.min_freq = next_existing.unwrap_or(0);
            }
        }

        if !self.buckets.contains_key(&next_freq) {
            let prev = if bucket_empty {
                prev_freq
            } else {
                Some(current_freq)
            };
            let next = next_existing;
            self.insert_bucket(next_freq, prev, next);
        }

        if let Some(entry) = self.entries.get_mut(id) {
            entry.freq = next_freq;
            entry.last_epoch = self.epoch;
        }
        self.list_push_front(next_freq, id);
        if self.min_freq == 0 || next_freq < self.min_freq {
            self.min_freq = next_freq;
        }

        Some(next_freq)
    }

    /// Halves all frequencies (rounding down), clamping at 1.
    ///
    /// This is an O(n) rebuild and will reorder tie-breaks within buckets.
    /// Useful for preventing frequency inflation over time.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBuckets;
    ///
    /// let mut freq = FrequencyBuckets::new();
    /// freq.insert("a");
    /// freq.insert("b");
    ///
    /// // Build up frequencies
    /// for _ in 0..9 { freq.touch(&"a"); }  // "a" at freq=10
    /// for _ in 0..3 { freq.touch(&"b"); }  // "b" at freq=4
    ///
    /// freq.decay_halve();
    ///
    /// assert_eq!(freq.frequency(&"a"), Some(5));  // 10 / 2 = 5
    /// assert_eq!(freq.frequency(&"b"), Some(2));  // 4 / 2 = 2
    /// ```
    pub fn decay_halve(&mut self) {
        if self.is_empty() {
            return;
        }
        self.rebuild_with(|freq| (freq / 2).max(1));
    }

    /// Rebases frequencies so the current minimum becomes 1.
    ///
    /// This is an O(n) rebuild and will reorder tie-breaks within buckets.
    /// Useful after evicting low-frequency items to reclaim frequency space.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBuckets;
    ///
    /// let mut freq = FrequencyBuckets::new();
    /// freq.insert("a");
    /// freq.insert("b");
    ///
    /// for _ in 0..4 { freq.touch(&"a"); }  // "a" at freq=5
    /// for _ in 0..2 { freq.touch(&"b"); }  // "b" at freq=3
    ///
    /// assert_eq!(freq.min_freq(), Some(3));
    ///
    /// freq.rebase_min_freq();
    ///
    /// // min_freq rebased to 1: subtract (3-1)=2 from all
    /// assert_eq!(freq.frequency(&"a"), Some(3));  // 5 - 2 = 3
    /// assert_eq!(freq.frequency(&"b"), Some(1));  // 3 - 2 = 1
    /// assert_eq!(freq.min_freq(), Some(1));
    /// ```
    pub fn rebase_min_freq(&mut self) {
        if self.min_freq <= 1 {
            return;
        }
        let delta = self.min_freq - 1;
        self.rebuild_with(|freq| freq.saturating_sub(delta).max(1));
    }

    /// Removes `key` from tracking and returns its previous frequency.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBuckets;
    ///
    /// let mut freq = FrequencyBuckets::new();
    /// freq.insert("key");
    /// freq.touch(&"key");
    ///
    /// assert_eq!(freq.remove(&"key"), Some(2));
    /// assert_eq!(freq.remove(&"key"), None);  // Already removed
    /// assert!(!freq.contains(&"key"));
    /// ```
    #[inline]
    pub fn remove(&mut self, key: &K) -> Option<u64> {
        let id = self.index.remove(key)?;
        let freq = self.entries.get(id)?.freq;

        self.list_remove(freq, id)?;
        let bucket_empty = self.bucket_is_empty(freq);
        let (prev, next) = {
            let bucket = self.buckets.get(&freq)?;
            (bucket.prev, bucket.next)
        };

        if bucket_empty {
            self.remove_bucket(freq, prev, next);
            if self.min_freq == freq {
                self.min_freq = next.unwrap_or(0);
            }
        }

        self.entries.remove(id).map(|entry| entry.freq)
    }

    /// Removes a batch of keys; returns number of keys removed.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBuckets;
    ///
    /// let mut freq = FrequencyBuckets::new();
    /// freq.insert_batch(["a", "b", "c"]);
    ///
    /// let removed = freq.remove_batch(["a", "c", "missing"]);
    /// assert_eq!(removed, 2);  // Only "a" and "c" found
    /// assert_eq!(freq.len(), 1);
    /// ```
    pub fn remove_batch<I>(&mut self, keys: I) -> usize
    where
        I: IntoIterator<Item = K>,
    {
        let mut removed = 0;
        for key in keys {
            if self.remove(&key).is_some() {
                removed += 1;
            }
        }
        removed
    }

    /// Removes and returns the eviction candidate `(key, freq)`.
    ///
    /// Eviction is O(1) using `min_freq` and the tail of that bucket.
    /// Within a frequency bucket, FIFO ordering is used (oldest evicted first).
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBuckets;
    ///
    /// let mut freq = FrequencyBuckets::new();
    /// freq.insert("a");
    /// freq.insert("b");
    /// freq.insert("c");
    /// freq.touch(&"c");  // "c" at freq=2
    ///
    /// // Evict in FIFO order at min frequency
    /// assert_eq!(freq.pop_min(), Some(("a", 1)));  // First inserted at freq=1
    /// assert_eq!(freq.pop_min(), Some(("b", 1)));  // Second inserted at freq=1
    /// assert_eq!(freq.pop_min(), Some(("c", 2)));  // Only one left
    /// assert_eq!(freq.pop_min(), None);            // Empty
    /// ```
    #[inline]
    pub fn pop_min(&mut self) -> Option<(K, u64)> {
        let freq = self.min_freq;
        if freq == 0 {
            return None;
        }

        let id = self.buckets.get(&freq)?.tail?;
        self.list_remove(freq, id)?;
        let bucket_empty = self.bucket_is_empty(freq);
        let (prev, next) = {
            let bucket = self.buckets.get(&freq)?;
            (bucket.prev, bucket.next)
        };

        if bucket_empty {
            self.remove_bucket(freq, prev, next);
            if self.min_freq == freq {
                self.min_freq = next.unwrap_or(0);
            }
        }

        let entry = self.entries.remove(id)?;
        self.index.remove(&entry.key);
        Some((entry.key, entry.freq))
    }

    /// Clears all state.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBuckets;
    ///
    /// let mut freq = FrequencyBuckets::new();
    /// freq.insert("a");
    /// freq.insert("b");
    ///
    /// freq.clear();
    /// assert!(freq.is_empty());
    /// assert_eq!(freq.min_freq(), None);
    /// ```
    pub fn clear(&mut self) {
        self.entries.clear();
        self.index.clear();
        self.buckets.clear();
        self.min_freq = 0;
        self.epoch = 0;
    }

    /// Clears all state and shrinks internal storage.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBuckets;
    ///
    /// let mut freq = FrequencyBuckets::with_capacity(100);
    /// freq.insert("a");
    ///
    /// freq.clear_shrink();
    /// assert!(freq.is_empty());
    /// ```
    pub fn clear_shrink(&mut self) {
        self.clear();
        self.entries.shrink_to_fit();
        self.index.shrink_to_fit();
        self.buckets.shrink_to_fit();
    }

    /// Returns an approximate memory footprint in bytes.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBuckets;
    ///
    /// let freq: FrequencyBuckets<u64> = FrequencyBuckets::new();
    /// let bytes = freq.approx_bytes();
    /// assert!(bytes > 0);
    /// ```
    pub fn approx_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.entries.approx_bytes()
            + self.index.capacity() * std::mem::size_of::<(K, SlotId)>()
            + self.buckets.capacity() * std::mem::size_of::<(u64, Bucket)>()
    }

    fn ensure_bucket(&mut self, freq: u64) {
        if self.buckets.contains_key(&freq) {
            return;
        }
        let mut prev = None;
        let mut next = None;
        for &f in self.buckets.keys() {
            if f < freq && prev.is_none_or(|p| f > p) {
                prev = Some(f);
            }
            if f > freq && next.is_none_or(|n| f < n) {
                next = Some(f);
            }
        }
        self.insert_bucket(freq, prev, next);
    }

    fn insert_with_freq(&mut self, key: K, freq: u64) {
        let freq = freq.max(1);
        let id = self.entries.insert(Entry {
            key: key.clone(),
            freq,
            last_epoch: self.epoch,
            prev: None,
            next: None,
        });
        self.index.insert(key, id);
        self.ensure_bucket(freq);
        self.list_push_front(freq, id);
        if self.min_freq == 0 || freq < self.min_freq {
            self.min_freq = freq;
        }
    }

    fn rebuild_with<F>(&mut self, mut f: F)
    where
        F: FnMut(u64) -> u64,
    {
        let entries: Vec<(K, u64)> = self
            .entries
            .iter()
            .map(|(_, entry)| (entry.key.clone(), f(entry.freq)))
            .collect();
        self.entries.clear();
        self.index.clear();
        self.buckets.clear();
        self.min_freq = 0;

        for (key, freq) in entries {
            self.insert_with_freq(key, freq);
        }
    }

    #[cfg(any(test, debug_assertions))]
    /// Returns a debug snapshot of bucket chains.
    pub fn debug_snapshot(&self) -> FrequencyBucketsSnapshot<K> {
        let mut buckets: Vec<(u64, Vec<SlotId>)> = self
            .buckets
            .keys()
            .copied()
            .map(|freq| (freq, self.iter_bucket_ids(freq).collect()))
            .collect();
        for (_, ids) in &mut buckets {
            ids.sort_by_key(|id| id.index());
        }
        buckets.sort_by_key(|(freq, _)| *freq);
        let mut bucket_entries: Vec<(u64, Vec<FrequencyBucketEntryDebug<K>>)> = self
            .buckets
            .keys()
            .copied()
            .map(|freq| {
                let mut entries: Vec<_> = self
                    .iter_bucket_entries(freq)
                    .map(|(id, meta)| FrequencyBucketEntryDebug {
                        id,
                        key: meta.key.clone(),
                        freq: meta.freq,
                        last_epoch: meta.last_epoch,
                    })
                    .collect();
                entries.sort_by_key(|entry| entry.id.index());
                (freq, entries)
            })
            .collect();
        bucket_entries.sort_by_key(|(freq, _)| *freq);
        let mut entry_epochs: Vec<(SlotId, u64)> = self
            .entries
            .iter()
            .map(|(id, entry)| (id, entry.last_epoch))
            .collect();
        entry_epochs.sort_by_key(|(id, _)| id.index());
        FrequencyBucketsSnapshot {
            min_freq: self.min_freq(),
            entries_len: self.entries.len(),
            index_len: self.index.len(),
            buckets,
            epoch: self.epoch,
            entry_epochs,
            bucket_entries,
        }
    }

    #[cfg(any(test, debug_assertions))]
    pub fn debug_validate_invariants(&self) {
        assert_eq!(self.len(), self.index.len());

        if self.is_empty() {
            assert!(self.buckets.is_empty());
            assert_eq!(self.min_freq, 0);
            return;
        }

        assert!(self.min_freq > 0);
        assert!(self.buckets.contains_key(&self.min_freq));

        for (&freq, bucket) in &self.buckets {
            assert!(bucket.head.is_some());
            assert!(bucket.tail.is_some());
            if let Some(prev) = bucket.prev {
                assert!(self.buckets.contains_key(&prev));
                assert_eq!(self.buckets[&prev].next, Some(freq));
            } else {
                assert_eq!(self.min_freq, freq);
            }
            if let Some(next) = bucket.next {
                assert!(self.buckets.contains_key(&next));
                assert_eq!(self.buckets[&next].prev, Some(freq));
            }

            let mut current = bucket.head;
            let mut last = None;
            let mut count = 0usize;
            while let Some(id) = current {
                let entry = self.entries.get(id).expect("bucket entry missing");
                assert_eq!(entry.freq, freq);
                assert_eq!(entry.prev, last);
                assert_eq!(self.index.get(&entry.key), Some(&id));
                last = Some(id);
                current = entry.next;
                count += 1;
            }
            assert_eq!(bucket.tail, last);
            assert!(count > 0);
        }
    }

    fn bucket_is_empty(&self, freq: u64) -> bool {
        self.buckets
            .get(&freq)
            .map(|bucket| bucket.head.is_none())
            .unwrap_or(true)
    }

    fn insert_bucket(&mut self, freq: u64, prev: Option<u64>, next: Option<u64>) {
        let bucket = Bucket {
            head: None,
            tail: None,
            prev,
            next,
        };
        self.buckets.insert(freq, bucket);

        if let Some(prev) = prev {
            if let Some(prev_bucket) = self.buckets.get_mut(&prev) {
                prev_bucket.next = Some(freq);
            }
        }
        if let Some(next) = next {
            if let Some(next_bucket) = self.buckets.get_mut(&next) {
                next_bucket.prev = Some(freq);
            }
        }
    }

    fn remove_bucket(&mut self, freq: u64, prev: Option<u64>, next: Option<u64>) {
        if let Some(prev) = prev {
            if let Some(prev_bucket) = self.buckets.get_mut(&prev) {
                prev_bucket.next = next;
            }
        }
        if let Some(next) = next {
            if let Some(next_bucket) = self.buckets.get_mut(&next) {
                next_bucket.prev = prev;
            }
        }
        self.buckets.remove(&freq);
    }

    fn list_push_front(&mut self, freq: u64, id: SlotId) {
        let bucket = self.buckets.get_mut(&freq).expect("bucket missing");

        let old_head = bucket.head;
        if let Some(entry) = self.entries.get_mut(id) {
            entry.prev = None;
            entry.next = old_head;
        }
        if let Some(old_head) = old_head {
            if let Some(entry) = self.entries.get_mut(old_head) {
                entry.prev = Some(id);
            }
        } else {
            bucket.tail = Some(id);
        }
        bucket.head = Some(id);
    }

    fn list_remove(&mut self, freq: u64, id: SlotId) -> Option<()> {
        let (prev, next) = {
            let entry = self.entries.get(id)?;
            (entry.prev, entry.next)
        };

        let bucket = self.buckets.get_mut(&freq)?;
        if let Some(prev) = prev {
            if let Some(entry) = self.entries.get_mut(prev) {
                entry.next = next;
            }
        } else {
            bucket.head = next;
        }
        if let Some(next) = next {
            if let Some(entry) = self.entries.get_mut(next) {
                entry.prev = prev;
            }
        } else {
            bucket.tail = prev;
        }

        if let Some(entry) = self.entries.get_mut(id) {
            entry.prev = None;
            entry.next = None;
        }

        Some(())
    }
}

#[cfg(any(test, debug_assertions))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrequencyBucketsSnapshot<K> {
    pub min_freq: Option<u64>,
    pub entries_len: usize,
    pub index_len: usize,
    pub buckets: Vec<(u64, Vec<SlotId>)>,
    pub epoch: u64,
    pub entry_epochs: Vec<(SlotId, u64)>,
    pub bucket_entries: Vec<(u64, Vec<FrequencyBucketEntryDebug<K>>)>,
}

/// Sharded frequency buckets for reduced contention.
///
/// Distributes keys across multiple shards, each protected by its own `RwLock`.
/// Reduces lock contention in concurrent workloads at the cost of approximate
/// global LFU (eviction selects from the shard with the lowest `min_freq`).
///
/// # Type Parameters
///
/// - `K`: Key type, must be `Eq + Hash + Clone`
///
/// # Example
///
/// ```
/// use cachekit::ds::ShardedFrequencyBuckets;
///
/// let freq = ShardedFrequencyBuckets::new(4);  // 4 shards
///
/// freq.insert("a");
/// freq.insert("b");
/// freq.touch(&"a");
///
/// assert_eq!(freq.frequency(&"a"), Some(2));
/// assert_eq!(freq.len(), 2);
/// ```
///
/// # Multi-threaded Example
///
/// ```
/// use std::sync::Arc;
/// use std::thread;
/// use cachekit::ds::ShardedFrequencyBuckets;
///
/// let freq = Arc::new(ShardedFrequencyBuckets::new(4));
///
/// let handles: Vec<_> = (0..4).map(|t| {
///     let freq = Arc::clone(&freq);
///     thread::spawn(move || {
///         for i in 0..25 {
///             let key = format!("key_{}_{}", t, i);
///             freq.insert(key.clone());
///             freq.touch(&key);
///         }
///     })
/// }).collect();
///
/// for h in handles {
///     h.join().unwrap();
/// }
///
/// assert_eq!(freq.len(), 100);
/// ```
#[cfg(feature = "concurrency")]
#[derive(Debug)]
pub struct ShardedFrequencyBuckets<K> {
    shards: Vec<parking_lot::RwLock<FrequencyBuckets<K>>>,
    selector: crate::ds::ShardSelector,
}

#[cfg(feature = "concurrency")]
impl<K> ShardedFrequencyBuckets<K>
where
    K: Eq + Hash + Clone,
{
    /// Creates a sharded tracker with `shards` shards.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ShardedFrequencyBuckets;
    ///
    /// let freq: ShardedFrequencyBuckets<String> = ShardedFrequencyBuckets::new(4);
    /// assert_eq!(freq.shard_count(), 4);
    /// assert!(freq.is_empty());
    /// ```
    pub fn new(shards: usize) -> Self {
        let shards = shards.max(1);
        let mut vec = Vec::with_capacity(shards);
        for _ in 0..shards {
            vec.push(parking_lot::RwLock::new(FrequencyBuckets::new()));
        }
        Self {
            shards: vec,
            selector: crate::ds::ShardSelector::new(shards, 0),
        }
    }

    /// Creates a sharded tracker with `shards` and `capacity_per_shard`.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ShardedFrequencyBuckets;
    ///
    /// let freq: ShardedFrequencyBuckets<String> = ShardedFrequencyBuckets::with_shards(4, 1000);
    /// assert_eq!(freq.shard_count(), 4);
    /// ```
    pub fn with_shards(shards: usize, capacity_per_shard: usize) -> Self {
        let shards = shards.max(1);
        let mut vec = Vec::with_capacity(shards);
        for _ in 0..shards {
            vec.push(parking_lot::RwLock::new(FrequencyBuckets::with_capacity(
                capacity_per_shard,
            )));
        }
        Self {
            shards: vec,
            selector: crate::ds::ShardSelector::new(shards, 0),
        }
    }

    /// Creates a sharded tracker with `shards`, `capacity_per_shard`, and a hash seed.
    ///
    /// The seed allows deterministic shard assignment for testing.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ShardedFrequencyBuckets;
    ///
    /// let freq: ShardedFrequencyBuckets<String> =
    ///     ShardedFrequencyBuckets::with_shards_seed(4, 100, 42);
    /// assert_eq!(freq.shard_count(), 4);
    /// ```
    pub fn with_shards_seed(shards: usize, capacity_per_shard: usize, seed: u64) -> Self {
        let shards = shards.max(1);
        let mut vec = Vec::with_capacity(shards);
        for _ in 0..shards {
            vec.push(parking_lot::RwLock::new(FrequencyBuckets::with_capacity(
                capacity_per_shard,
            )));
        }
        Self {
            shards: vec,
            selector: crate::ds::ShardSelector::new(shards, seed),
        }
    }

    /// Returns the number of shards.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ShardedFrequencyBuckets;
    ///
    /// let freq: ShardedFrequencyBuckets<&str> = ShardedFrequencyBuckets::new(8);
    /// assert_eq!(freq.shard_count(), 8);
    /// ```
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    fn shard_for(&self, key: &K) -> usize {
        self.selector.shard_for_key(key)
    }

    /// Returns the shard index for `key` using the configured selector.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ShardedFrequencyBuckets;
    ///
    /// let freq: ShardedFrequencyBuckets<&str> = ShardedFrequencyBuckets::new(4);
    /// let shard = freq.shard_for_key(&"my_key");
    /// assert!(shard < 4);
    /// ```
    pub fn shard_for_key(&self, key: &K) -> usize {
        self.selector.shard_for_key(key)
    }

    /// Inserts a key into its shard.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ShardedFrequencyBuckets;
    ///
    /// let freq = ShardedFrequencyBuckets::new(4);
    /// assert!(freq.insert("key"));
    /// assert!(!freq.insert("key"));  // Already exists
    /// ```
    pub fn insert(&self, key: K) -> bool {
        let shard = self.shard_for(&key);
        let mut buckets = self.shards[shard].write();
        buckets.insert(key)
    }

    /// Touches a key in its shard.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ShardedFrequencyBuckets;
    ///
    /// let freq = ShardedFrequencyBuckets::new(4);
    /// freq.insert("key");
    ///
    /// assert_eq!(freq.touch(&"key"), Some(2));
    /// assert_eq!(freq.touch(&"missing"), None);
    /// ```
    pub fn touch(&self, key: &K) -> Option<u64> {
        let shard = self.shard_for(key);
        let mut buckets = self.shards[shard].write();
        buckets.touch(key)
    }

    /// Removes a key from its shard.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ShardedFrequencyBuckets;
    ///
    /// let freq = ShardedFrequencyBuckets::new(4);
    /// freq.insert("key");
    ///
    /// assert_eq!(freq.remove(&"key"), Some(1));
    /// assert_eq!(freq.remove(&"key"), None);
    /// ```
    pub fn remove(&self, key: &K) -> Option<u64> {
        let shard = self.shard_for(key);
        let mut buckets = self.shards[shard].write();
        buckets.remove(key)
    }

    /// Returns the frequency for a key in its shard.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ShardedFrequencyBuckets;
    ///
    /// let freq = ShardedFrequencyBuckets::new(4);
    /// freq.insert("key");
    /// freq.touch(&"key");
    ///
    /// assert_eq!(freq.frequency(&"key"), Some(2));
    /// ```
    pub fn frequency(&self, key: &K) -> Option<u64> {
        let shard = self.shard_for(key);
        let buckets = self.shards[shard].read();
        buckets.frequency(key)
    }

    /// Returns `true` if the key exists in its shard.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ShardedFrequencyBuckets;
    ///
    /// let freq = ShardedFrequencyBuckets::new(4);
    /// freq.insert("key");
    ///
    /// assert!(freq.contains(&"key"));
    /// assert!(!freq.contains(&"missing"));
    /// ```
    pub fn contains(&self, key: &K) -> bool {
        let shard = self.shard_for(key);
        let buckets = self.shards[shard].read();
        buckets.contains(key)
    }

    /// Returns the total number of keys across all shards.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ShardedFrequencyBuckets;
    ///
    /// let freq = ShardedFrequencyBuckets::new(4);
    /// freq.insert("a");
    /// freq.insert("b");
    /// freq.insert("c");
    ///
    /// assert_eq!(freq.len(), 3);
    /// ```
    pub fn len(&self) -> usize {
        self.shards.iter().map(|b| b.read().len()).sum()
    }

    /// Returns `true` if all shards are empty.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ShardedFrequencyBuckets;
    ///
    /// let freq: ShardedFrequencyBuckets<&str> = ShardedFrequencyBuckets::new(4);
    /// assert!(freq.is_empty());
    ///
    /// freq.insert("key");
    /// assert!(!freq.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clears all shards.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ShardedFrequencyBuckets;
    ///
    /// let freq = ShardedFrequencyBuckets::new(4);
    /// freq.insert("a");
    /// freq.insert("b");
    ///
    /// freq.clear();
    /// assert!(freq.is_empty());
    /// ```
    pub fn clear(&self) {
        for shard in &self.shards {
            shard.write().clear();
        }
    }

    /// Clears all shards and shrinks internal storage.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ShardedFrequencyBuckets;
    ///
    /// let freq = ShardedFrequencyBuckets::with_shards(4, 100);
    /// freq.insert("a");
    ///
    /// freq.clear_shrink();
    /// assert!(freq.is_empty());
    /// ```
    pub fn clear_shrink(&self) {
        for shard in &self.shards {
            shard.write().clear_shrink();
        }
    }

    /// Returns a snapshot of `(SlotId, meta)` for a given frequency.
    ///
    /// Collects entries from all shards at the specified frequency.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ShardedFrequencyBuckets;
    ///
    /// let freq = ShardedFrequencyBuckets::new(4);
    /// freq.insert("a");
    /// freq.insert("b");
    /// freq.touch(&"a");  // "a" at freq=2
    ///
    /// // Get all entries at frequency 1
    /// let at_freq_1 = freq.iter_bucket_entries(1);
    /// assert_eq!(at_freq_1.len(), 1);
    /// assert_eq!(at_freq_1[0].1.key, "b");
    /// ```
    pub fn iter_bucket_entries(
        &self,
        freq: u64,
    ) -> Vec<(SlotId, ShardedFrequencyBucketEntryMeta<K>)> {
        let mut entries = Vec::new();
        for shard in &self.shards {
            let buckets = shard.read();
            entries.extend(buckets.iter_bucket_entries(freq).map(|(id, meta)| {
                (
                    id,
                    ShardedFrequencyBucketEntryMeta {
                        key: meta.key.clone(),
                        freq: meta.freq,
                        last_epoch: meta.last_epoch,
                    },
                )
            }));
        }
        entries
    }

    /// Returns a snapshot of all `(SlotId, meta)` entries across all shards.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ShardedFrequencyBuckets;
    ///
    /// let freq = ShardedFrequencyBuckets::new(4);
    /// freq.insert("a");
    /// freq.insert("b");
    /// freq.touch(&"a");
    ///
    /// let all = freq.iter_entries();
    /// assert_eq!(all.len(), 2);
    /// ```
    pub fn iter_entries(&self) -> Vec<(SlotId, ShardedFrequencyBucketEntryMeta<K>)> {
        let mut entries = Vec::new();
        for shard in &self.shards {
            let buckets = shard.read();
            entries.extend(buckets.iter_entries().map(|(id, meta)| {
                (
                    id,
                    ShardedFrequencyBucketEntryMeta {
                        key: meta.key.clone(),
                        freq: meta.freq,
                        last_epoch: meta.last_epoch,
                    },
                )
            }));
        }
        entries
    }

    /// Returns a snapshot of `(shard_idx, SlotId, meta)` for a given frequency.
    ///
    /// Includes the shard index for each entry.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ShardedFrequencyBuckets;
    ///
    /// let freq = ShardedFrequencyBuckets::new(4);
    /// freq.insert("a");
    /// freq.insert("b");
    ///
    /// let entries = freq.iter_bucket_entries_with_shard(1);
    /// assert_eq!(entries.len(), 2);
    ///
    /// // Each entry includes its shard index
    /// for (shard_idx, _slot_id, meta) in &entries {
    ///     assert!(*shard_idx < 4);
    ///     assert_eq!(meta.freq, 1);
    /// }
    /// ```
    pub fn iter_bucket_entries_with_shard(
        &self,
        freq: u64,
    ) -> Vec<(usize, SlotId, ShardedFrequencyBucketEntryMeta<K>)> {
        let mut entries = Vec::new();
        for (idx, shard) in self.shards.iter().enumerate() {
            let buckets = shard.read();
            entries.extend(buckets.iter_bucket_entries(freq).map(|(id, meta)| {
                (
                    idx,
                    id,
                    ShardedFrequencyBucketEntryMeta {
                        key: meta.key.clone(),
                        freq: meta.freq,
                        last_epoch: meta.last_epoch,
                    },
                )
            }));
        }
        entries
    }

    /// Returns a snapshot of all `(shard_idx, SlotId, meta)` entries.
    ///
    /// Includes the shard index for each entry.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ShardedFrequencyBuckets;
    ///
    /// let freq = ShardedFrequencyBuckets::new(4);
    /// freq.insert("a");
    /// freq.insert("b");
    /// freq.touch(&"a");
    ///
    /// let all = freq.iter_entries_with_shard();
    /// assert_eq!(all.len(), 2);
    ///
    /// // Find the entry for "a" and verify its frequency
    /// let a_entry = all.iter().find(|(_, _, m)| m.key == "a").unwrap();
    /// assert_eq!(a_entry.2.freq, 2);
    /// ```
    pub fn iter_entries_with_shard(
        &self,
    ) -> Vec<(usize, SlotId, ShardedFrequencyBucketEntryMeta<K>)> {
        let mut entries = Vec::new();
        for (idx, shard) in self.shards.iter().enumerate() {
            let buckets = shard.read();
            entries.extend(buckets.iter_entries().map(|(id, meta)| {
                (
                    idx,
                    id,
                    ShardedFrequencyBucketEntryMeta {
                        key: meta.key.clone(),
                        freq: meta.freq,
                        last_epoch: meta.last_epoch,
                    },
                )
            }));
        }
        entries
    }

    /// Returns an approximate memory footprint in bytes.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ShardedFrequencyBuckets;
    ///
    /// let freq: ShardedFrequencyBuckets<u64> = ShardedFrequencyBuckets::new(4);
    /// let bytes = freq.approx_bytes();
    /// assert!(bytes > 0);
    /// ```
    pub fn approx_bytes(&self) -> usize {
        self.shards
            .iter()
            .map(|shard| shard.read().approx_bytes())
            .sum()
    }

    /// Peeks the global min across shards by cloning the candidate.
    ///
    /// Scans all shards to find the one with the lowest `min_freq`.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ShardedFrequencyBuckets;
    ///
    /// let freq = ShardedFrequencyBuckets::new(4);
    /// freq.insert("a");
    /// freq.insert("b");
    /// freq.touch(&"b");  // "b" at freq=2
    ///
    /// // Peek returns "a" (lowest frequency)
    /// let (key, freq_val) = freq.peek_min().unwrap();
    /// assert_eq!(key, "a");
    /// assert_eq!(freq_val, 1);
    /// assert_eq!(freq.len(), 2);  // Not removed
    /// ```
    pub fn peek_min(&self) -> Option<(K, u64)> {
        let mut best: Option<(usize, u64)> = None;
        for (idx, shard) in self.shards.iter().enumerate() {
            let buckets = shard.read();
            let Some(freq) = buckets.min_freq() else {
                continue;
            };
            let is_better = best.is_none_or(|(_, best_freq)| freq < best_freq);
            if is_better {
                best = Some((idx, freq));
            }
        }
        let (idx, _) = best?;
        let buckets = self.shards[idx].read();
        buckets.peek_min().map(|(key, freq)| (key.clone(), freq))
    }

    /// Pops the global min across shards (scan by min_freq).
    ///
    /// Scans all shards, finds the one with the lowest `min_freq`, and
    /// evicts from that shard.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ShardedFrequencyBuckets;
    ///
    /// let freq = ShardedFrequencyBuckets::new(4);
    /// freq.insert("a");
    /// freq.insert("b");
    /// freq.touch(&"b");  // "b" at freq=2
    ///
    /// // Pop evicts "a" (lowest frequency)
    /// let (key, freq_val) = freq.pop_min().unwrap();
    /// assert_eq!(key, "a");
    /// assert_eq!(freq_val, 1);
    /// assert_eq!(freq.len(), 1);
    /// ```
    pub fn pop_min(&self) -> Option<(K, u64)> {
        let mut best: Option<(usize, u64)> = None;
        for (idx, shard) in self.shards.iter().enumerate() {
            let buckets = shard.read();
            let Some(freq) = buckets.min_freq() else {
                continue;
            };
            let is_better = best.is_none_or(|(_, best_freq)| freq < best_freq);
            if is_better {
                best = Some((idx, freq));
            }
        }
        let (idx, _) = best?;
        let mut buckets = self.shards[idx].write();
        buckets.pop_min()
    }
}

/// Frequency buckets keyed by a compact handle (for interned keys).
///
/// Wraps [`FrequencyBuckets`] for use with interned keys where `H: Copy`.
/// Avoids cloning large keys by using compact handles (e.g., `u64`) that
/// are interned elsewhere (see [`KeyInterner`](crate::ds::KeyInterner)).
///
/// # Type Parameters
///
/// - `H`: Handle type, must be `Eq + Hash + Copy`
///
/// # Example
///
/// ```
/// use cachekit::ds::{FrequencyBucketsHandle, KeyInterner};
///
/// // Use KeyInterner to map string keys to u64 handles
/// let mut interner = KeyInterner::new();
/// let mut freq = FrequencyBucketsHandle::new();
///
/// let h1 = interner.intern(&"long_key_name".to_string());
/// let h2 = interner.intern(&"another_key".to_string());
///
/// freq.insert(h1);
/// freq.insert(h2);
/// freq.touch(&h1);
///
/// assert_eq!(freq.frequency(&h1), Some(2));
/// assert_eq!(freq.frequency(&h2), Some(1));
///
/// // Evict LFU (handle h2)
/// let (evicted_handle, freq_val) = freq.pop_min().unwrap();
/// assert_eq!(evicted_handle, h2);
///
/// // Resolve handle back to key if needed
/// assert_eq!(interner.resolve(evicted_handle), Some(&"another_key".to_string()));
/// ```
#[derive(Debug)]
pub struct FrequencyBucketsHandle<H> {
    inner: FrequencyBuckets<H>,
}

impl<H> FrequencyBucketsHandle<H>
where
    H: Eq + Hash + Copy,
{
    /// Creates an empty tracker.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBucketsHandle;
    ///
    /// let freq: FrequencyBucketsHandle<u64> = FrequencyBucketsHandle::new();
    /// assert!(freq.is_empty());
    /// ```
    pub fn new() -> Self {
        Self {
            inner: FrequencyBuckets::new(),
        }
    }

    /// Creates a tracker with pre-allocated capacity.
    ///
    /// Uses [`DEFAULT_BUCKET_PREALLOC`] for the bucket map. For custom bucket
    /// pre-allocation, use [`with_capacity_and_bucket_hint`](Self::with_capacity_and_bucket_hint).
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBucketsHandle;
    ///
    /// let freq: FrequencyBucketsHandle<u64> = FrequencyBucketsHandle::with_capacity(1000);
    /// assert!(freq.is_empty());
    /// ```
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: FrequencyBuckets::with_capacity(capacity),
        }
    }

    /// Creates a tracker with pre-allocated capacity and custom bucket pre-allocation.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Pre-allocated space for entries and index
    /// * `bucket_hint` - Pre-allocated space for frequency buckets (number of distinct frequencies)
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBucketsHandle;
    ///
    /// // Expect many distinct frequencies
    /// let freq: FrequencyBucketsHandle<u64> = FrequencyBucketsHandle::with_capacity_and_bucket_hint(1000, 64);
    /// assert!(freq.is_empty());
    /// ```
    pub fn with_capacity_and_bucket_hint(capacity: usize, bucket_hint: usize) -> Self {
        Self {
            inner: FrequencyBuckets::with_capacity_and_bucket_hint(capacity, bucket_hint),
        }
    }

    /// Returns the number of tracked handles.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBucketsHandle;
    ///
    /// let mut freq = FrequencyBucketsHandle::new();
    /// assert_eq!(freq.len(), 0);
    ///
    /// freq.insert(1u64);
    /// freq.insert(2u64);
    /// assert_eq!(freq.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if there are no tracked handles.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBucketsHandle;
    ///
    /// let mut freq: FrequencyBucketsHandle<u64> = FrequencyBucketsHandle::new();
    /// assert!(freq.is_empty());
    ///
    /// freq.insert(1);
    /// assert!(!freq.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns `true` if `handle` is present.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBucketsHandle;
    ///
    /// let mut freq = FrequencyBucketsHandle::new();
    /// freq.insert(42u64);
    ///
    /// assert!(freq.contains(&42));
    /// assert!(!freq.contains(&99));
    /// ```
    pub fn contains(&self, handle: &H) -> bool {
        self.inner.contains(handle)
    }

    /// Returns the current frequency for `handle`, if present.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBucketsHandle;
    ///
    /// let mut freq = FrequencyBucketsHandle::new();
    /// freq.insert(1u64);
    /// freq.touch(&1);
    ///
    /// assert_eq!(freq.frequency(&1), Some(2));
    /// assert_eq!(freq.frequency(&99), None);
    /// ```
    pub fn frequency(&self, handle: &H) -> Option<u64> {
        self.inner.frequency(handle)
    }

    /// Returns the current epoch.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBucketsHandle;
    ///
    /// let freq: FrequencyBucketsHandle<u64> = FrequencyBucketsHandle::new();
    /// assert_eq!(freq.current_epoch(), 0);
    /// ```
    pub fn current_epoch(&self) -> u64 {
        self.inner.current_epoch()
    }

    /// Advances the epoch counter and returns the new value.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBucketsHandle;
    ///
    /// let mut freq: FrequencyBucketsHandle<u64> = FrequencyBucketsHandle::new();
    /// assert_eq!(freq.advance_epoch(), 1);
    /// assert_eq!(freq.advance_epoch(), 2);
    /// assert_eq!(freq.current_epoch(), 2);
    /// ```
    pub fn advance_epoch(&mut self) -> u64 {
        self.inner.advance_epoch()
    }

    /// Sets the epoch counter.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBucketsHandle;
    ///
    /// let mut freq: FrequencyBucketsHandle<u64> = FrequencyBucketsHandle::new();
    /// freq.set_epoch(100);
    /// assert_eq!(freq.current_epoch(), 100);
    /// ```
    pub fn set_epoch(&mut self, epoch: u64) {
        self.inner.set_epoch(epoch);
    }

    /// Returns the last epoch recorded for `handle`.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBucketsHandle;
    ///
    /// let mut freq = FrequencyBucketsHandle::new();
    /// freq.set_epoch(10);
    /// freq.insert(1u64);
    ///
    /// assert_eq!(freq.entry_epoch(&1), Some(10));
    /// ```
    pub fn entry_epoch(&self, handle: &H) -> Option<u64> {
        self.inner.entry_epoch(handle)
    }

    /// Sets the last epoch for `handle`; returns `false` if missing.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBucketsHandle;
    ///
    /// let mut freq = FrequencyBucketsHandle::new();
    /// freq.insert(1u64);
    ///
    /// assert!(freq.set_entry_epoch(&1, 42));
    /// assert_eq!(freq.entry_epoch(&1), Some(42));
    /// ```
    pub fn set_entry_epoch(&mut self, handle: &H, epoch: u64) -> bool {
        self.inner.set_entry_epoch(handle, epoch)
    }

    /// Returns the minimum frequency currently present.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBucketsHandle;
    ///
    /// let mut freq = FrequencyBucketsHandle::new();
    /// assert_eq!(freq.min_freq(), None);
    ///
    /// freq.insert(1u64);
    /// freq.insert(2u64);
    /// freq.touch(&1);
    ///
    /// assert_eq!(freq.min_freq(), Some(1));  // Handle 2 is at freq=1
    /// ```
    pub fn min_freq(&self) -> Option<u64> {
        self.inner.min_freq()
    }

    /// Peeks the eviction candidate `(handle, freq)`.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBucketsHandle;
    ///
    /// let mut freq = FrequencyBucketsHandle::new();
    /// freq.insert(1u64);
    /// freq.insert(2u64);
    /// freq.touch(&2);
    ///
    /// let (handle, freq_val) = freq.peek_min().unwrap();
    /// assert_eq!(handle, 1);  // Handle 1 has lowest freq
    /// assert_eq!(freq_val, 1);
    /// ```
    pub fn peek_min(&self) -> Option<(H, u64)> {
        self.inner.peek_min().map(|(handle, freq)| (*handle, freq))
    }

    /// Peeks the eviction candidate by reference `(handle, freq)`.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBucketsHandle;
    ///
    /// let mut freq = FrequencyBucketsHandle::new();
    /// freq.insert(1u64);
    ///
    /// let (handle_ref, freq_val) = freq.peek_min_ref().unwrap();
    /// assert_eq!(*handle_ref, 1);
    /// assert_eq!(freq_val, 1);
    /// ```
    pub fn peek_min_ref(&self) -> Option<(&H, u64)> {
        self.inner.peek_min()
    }

    /// Returns an iterator of SlotIds for a given frequency, from head to tail.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBucketsHandle;
    ///
    /// let mut freq = FrequencyBucketsHandle::new();
    /// freq.insert(1u64);
    /// freq.insert(2u64);
    ///
    /// let ids: Vec<_> = freq.iter_bucket_ids(1).collect();
    /// assert_eq!(ids.len(), 2);
    /// ```
    pub fn iter_bucket_ids(&self, freq: u64) -> FrequencyBucketIdIter<'_, H> {
        self.inner.iter_bucket_ids(freq)
    }

    /// Returns an iterator of `(SlotId, meta)` for a given frequency.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBucketsHandle;
    ///
    /// let mut freq = FrequencyBucketsHandle::new();
    /// freq.insert(1u64);
    /// freq.insert(2u64);
    /// freq.touch(&1);
    ///
    /// // Only handle 2 is at frequency 1
    /// let entries: Vec<_> = freq.iter_bucket_entries(1).collect();
    /// assert_eq!(entries.len(), 1);
    /// assert_eq!(*entries[0].1.key, 2);
    /// ```
    pub fn iter_bucket_entries(&self, freq: u64) -> FrequencyBucketEntryIter<'_, H> {
        self.inner.iter_bucket_entries(freq)
    }

    /// Returns an iterator over all `(SlotId, meta)` entries.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBucketsHandle;
    ///
    /// let mut freq = FrequencyBucketsHandle::new();
    /// freq.insert(1u64);
    /// freq.insert(2u64);
    ///
    /// let entries: Vec<_> = freq.iter_entries().collect();
    /// assert_eq!(entries.len(), 2);
    /// ```
    pub fn iter_entries(&self) -> impl Iterator<Item = (SlotId, FrequencyBucketEntryMeta<'_, H>)> {
        self.inner.iter_entries()
    }

    /// Inserts a new handle with frequency 1.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBucketsHandle;
    ///
    /// let mut freq = FrequencyBucketsHandle::new();
    ///
    /// assert!(freq.insert(1u64));   // New handle
    /// assert!(!freq.insert(1u64));  // Already exists
    /// assert_eq!(freq.frequency(&1), Some(1));
    /// ```
    pub fn insert(&mut self, handle: H) -> bool {
        self.inner.insert(handle)
    }

    /// Increments frequency for `handle` and returns the new frequency.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBucketsHandle;
    ///
    /// let mut freq = FrequencyBucketsHandle::new();
    /// freq.insert(1u64);
    ///
    /// assert_eq!(freq.touch(&1), Some(2));
    /// assert_eq!(freq.touch(&1), Some(3));
    /// assert_eq!(freq.touch(&99), None);  // Missing
    /// ```
    pub fn touch(&mut self, handle: &H) -> Option<u64> {
        self.inner.touch(handle)
    }

    /// Increments frequency for `handle`, clamping at `max_freq`.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBucketsHandle;
    ///
    /// let mut freq = FrequencyBucketsHandle::new();
    /// freq.insert(1u64);
    ///
    /// assert_eq!(freq.touch_capped(&1, 3), Some(2));
    /// assert_eq!(freq.touch_capped(&1, 3), Some(3));
    /// assert_eq!(freq.touch_capped(&1, 3), Some(3));  // Capped
    /// ```
    pub fn touch_capped(&mut self, handle: &H, max_freq: u64) -> Option<u64> {
        self.inner.touch_capped(handle, max_freq)
    }

    /// Removes `handle` from tracking and returns its previous frequency.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBucketsHandle;
    ///
    /// let mut freq = FrequencyBucketsHandle::new();
    /// freq.insert(1u64);
    /// freq.touch(&1);
    ///
    /// assert_eq!(freq.remove(&1), Some(2));
    /// assert_eq!(freq.remove(&1), None);  // Already removed
    /// ```
    pub fn remove(&mut self, handle: &H) -> Option<u64> {
        self.inner.remove(handle)
    }

    /// Removes and returns the eviction candidate `(handle, freq)`.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBucketsHandle;
    ///
    /// let mut freq = FrequencyBucketsHandle::new();
    /// freq.insert(1u64);
    /// freq.insert(2u64);
    /// freq.touch(&2);
    ///
    /// let (handle, freq_val) = freq.pop_min().unwrap();
    /// assert_eq!(handle, 1);  // Handle 1 evicted (lowest freq)
    /// assert_eq!(freq_val, 1);
    /// ```
    pub fn pop_min(&mut self) -> Option<(H, u64)> {
        self.inner.pop_min()
    }

    /// Halves all frequencies (rounding down), clamping at 1.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBucketsHandle;
    ///
    /// let mut freq = FrequencyBucketsHandle::new();
    /// freq.insert(1u64);
    /// for _ in 0..9 { freq.touch(&1); }  // freq=10
    ///
    /// freq.decay_halve();
    /// assert_eq!(freq.frequency(&1), Some(5));
    /// ```
    pub fn decay_halve(&mut self) {
        self.inner.decay_halve();
    }

    /// Rebases frequencies so the current minimum becomes 1.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBucketsHandle;
    ///
    /// let mut freq = FrequencyBucketsHandle::new();
    /// freq.insert(1u64);
    /// freq.insert(2u64);
    /// for _ in 0..4 { freq.touch(&1); }  // handle 1 at freq=5
    /// for _ in 0..2 { freq.touch(&2); }  // handle 2 at freq=3
    ///
    /// freq.rebase_min_freq();
    /// assert_eq!(freq.frequency(&2), Some(1));  // 3 - 2 = 1
    /// assert_eq!(freq.frequency(&1), Some(3));  // 5 - 2 = 3
    /// ```
    pub fn rebase_min_freq(&mut self) {
        self.inner.rebase_min_freq();
    }

    /// Clears all state.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBucketsHandle;
    ///
    /// let mut freq = FrequencyBucketsHandle::new();
    /// freq.insert(1u64);
    /// freq.insert(2u64);
    ///
    /// freq.clear();
    /// assert!(freq.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Clears all state and shrinks internal storage.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBucketsHandle;
    ///
    /// let mut freq = FrequencyBucketsHandle::new();
    /// freq.insert(1u64);
    /// freq.insert(2u64);
    ///
    /// freq.clear_shrink();
    /// assert!(freq.is_empty());
    /// ```
    pub fn clear_shrink(&mut self) {
        self.inner.clear_shrink();
    }

    /// Returns an approximate memory footprint in bytes.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FrequencyBucketsHandle;
    ///
    /// let freq: FrequencyBucketsHandle<u64> = FrequencyBucketsHandle::new();
    /// let bytes = freq.approx_bytes();
    /// assert!(bytes > 0);
    /// ```
    pub fn approx_bytes(&self) -> usize {
        self.inner.approx_bytes()
    }

    #[cfg(any(test, debug_assertions))]
    /// Returns a debug snapshot of bucket chains.
    pub fn debug_snapshot(&self) -> FrequencyBucketsSnapshot<H> {
        self.inner.debug_snapshot()
    }

    #[cfg(any(test, debug_assertions))]
    /// Validates internal invariants (debug/test only).
    pub fn debug_validate_invariants(&self) {
        self.inner.debug_validate_invariants();
    }
}

impl<H> Default for FrequencyBucketsHandle<H>
where
    H: Eq + Hash + Copy,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator over SlotIds for a given frequency bucket.
///
/// Created by [`FrequencyBuckets::iter_bucket_ids`]. Yields entries from
/// head (MRU) to tail (LRU) within a single frequency bucket.
pub struct FrequencyBucketIdIter<'a, K> {
    buckets: &'a FrequencyBuckets<K>,
    current: Option<SlotId>,
}

impl<'a, K> Iterator for FrequencyBucketIdIter<'a, K> {
    type Item = SlotId;

    fn next(&mut self) -> Option<Self::Item> {
        let id = self.current?;
        let entry = self.buckets.entries.get(id)?;
        self.current = entry.next;
        Some(id)
    }
}

/// Iterator over `(SlotId, meta)` for a given frequency bucket.
///
/// Created by [`FrequencyBuckets::iter_bucket_entries`]. Yields entries from
/// head (MRU) to tail (LRU) within a single frequency bucket, including
/// metadata like key, frequency, and last_epoch.
pub struct FrequencyBucketEntryIter<'a, K> {
    buckets: &'a FrequencyBuckets<K>,
    current: Option<SlotId>,
}

impl<'a, K> Iterator for FrequencyBucketEntryIter<'a, K> {
    type Item = (SlotId, FrequencyBucketEntryMeta<'a, K>);

    fn next(&mut self) -> Option<Self::Item> {
        let id = self.current?;
        let entry = self.buckets.entries.get(id)?;
        self.current = entry.next;
        Some((
            id,
            FrequencyBucketEntryMeta {
                key: &entry.key,
                freq: entry.freq,
                last_epoch: entry.last_epoch,
            },
        ))
    }
}

impl<K> Default for FrequencyBuckets<K>
where
    K: Eq + Hash + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frequency_buckets_basic_flow() {
        let mut buckets = FrequencyBuckets::new();
        assert!(buckets.insert("a"));
        assert!(buckets.insert("b"));

        assert_eq!(buckets.frequency(&"a"), Some(1));
        assert_eq!(buckets.min_freq(), Some(1));

        assert_eq!(buckets.touch(&"a"), Some(2));
        assert_eq!(buckets.frequency(&"a"), Some(2));
        assert_eq!(buckets.min_freq(), Some(1));

        let popped = buckets.pop_min();
        assert_eq!(popped, Some(("b", 1)));
        assert_eq!(buckets.min_freq(), Some(2));
    }

    #[test]
    fn frequency_buckets_duplicate_insert_is_noop() {
        let mut buckets = FrequencyBuckets::new();
        assert!(buckets.insert("a"));
        assert!(!buckets.insert("a"));
        assert_eq!(buckets.len(), 1);
        assert_eq!(buckets.frequency(&"a"), Some(1));
    }

    #[test]
    fn frequency_buckets_touch_missing_returns_none() {
        let mut buckets: FrequencyBuckets<&str> = FrequencyBuckets::new();
        assert_eq!(buckets.touch(&"missing"), None);
        assert_eq!(buckets.min_freq(), None);
        assert!(buckets.is_empty());
    }

    #[test]
    fn frequency_buckets_remove_updates_min_freq() {
        let mut buckets = FrequencyBuckets::new();
        buckets.insert("a");
        buckets.insert("b");
        buckets.touch(&"b");
        assert_eq!(buckets.min_freq(), Some(1));

        assert_eq!(buckets.remove(&"a"), Some(1));
        assert_eq!(buckets.min_freq(), Some(2));
        assert!(!buckets.contains(&"a"));
        assert!(buckets.contains(&"b"));
    }

    #[test]
    fn frequency_buckets_pop_min_on_empty() {
        let mut buckets: FrequencyBuckets<&str> = FrequencyBuckets::new();
        assert_eq!(buckets.pop_min(), None);
        assert_eq!(buckets.peek_min(), None);
        assert_eq!(buckets.min_freq(), None);
    }

    #[test]
    fn frequency_buckets_peek_min_does_not_remove() {
        let mut buckets = FrequencyBuckets::new();
        buckets.insert("a");
        buckets.insert("b");
        let peeked = buckets.peek_min();
        assert!(matches!(peeked, Some((&"a", 1)) | Some((&"b", 1))));
        assert_eq!(buckets.len(), 2);
        assert!(buckets.contains(&"a"));
        assert!(buckets.contains(&"b"));
    }

    #[test]
    fn frequency_buckets_fifo_within_same_frequency() {
        let mut buckets = FrequencyBuckets::new();
        buckets.insert("a");
        buckets.insert("b");
        buckets.insert("c");

        let first = buckets.pop_min();
        assert_eq!(first, Some(("a", 1)));
        let second = buckets.pop_min();
        assert_eq!(second, Some(("b", 1)));
        let third = buckets.pop_min();
        assert_eq!(third, Some(("c", 1)));
        assert!(buckets.is_empty());
    }

    #[test]
    fn frequency_buckets_min_freq_tracks_next_bucket() {
        let mut buckets = FrequencyBuckets::new();
        buckets.insert("a");
        buckets.insert("b");
        buckets.insert("c");

        buckets.touch(&"a");
        buckets.touch(&"a");
        assert_eq!(buckets.frequency(&"a"), Some(3));
        assert_eq!(buckets.min_freq(), Some(1));

        buckets.pop_min();
        buckets.pop_min();
        assert_eq!(buckets.min_freq(), Some(3));
        assert_eq!(buckets.peek_min(), Some((&"a", 3)));
    }

    #[test]
    fn frequency_buckets_clear_resets_state() {
        let mut buckets = FrequencyBuckets::new();
        buckets.insert("a");
        buckets.insert("b");
        buckets.touch(&"a");
        buckets.clear();
        assert!(buckets.is_empty());
        assert_eq!(buckets.min_freq(), None);
        assert_eq!(buckets.pop_min(), None);
        assert_eq!(buckets.peek_min(), None);
    }

    #[test]
    fn frequency_buckets_debug_invariants_hold() {
        let mut buckets = FrequencyBuckets::new();
        buckets.insert("a");
        buckets.insert("b");
        buckets.touch(&"a");
        buckets.touch(&"a");
        buckets.remove(&"b");
        buckets.debug_validate_invariants();
    }

    #[test]
    fn frequency_buckets_peek_min_id_and_key() {
        let mut buckets = FrequencyBuckets::new();
        buckets.insert("a");
        buckets.insert("b");
        let id = buckets.peek_min_id().unwrap();
        let key = buckets.peek_min_key().unwrap();
        assert!(matches!(key, &"a" | &"b"));
        assert_eq!(buckets.frequency(key), Some(1));

        let entry = buckets.entries.get(id).unwrap();
        assert_eq!(&entry.key, key);
    }

    #[test]
    fn frequency_buckets_iter_bucket_ids_order() {
        let mut buckets = FrequencyBuckets::new();
        buckets.insert("a");
        buckets.insert("b");
        buckets.insert("c");

        let ids: Vec<_> = buckets.iter_bucket_ids(1).collect();
        assert_eq!(ids.len(), 3);
        let keys: Vec<_> = ids
            .iter()
            .map(|id| buckets.entries.get(*id).unwrap().key)
            .collect();
        assert_eq!(keys, vec!["c", "b", "a"]);
    }

    #[test]
    fn frequency_buckets_iter_bucket_entries_meta() {
        let mut buckets = FrequencyBuckets::new();
        buckets.insert("a");
        buckets.insert("b");
        buckets.touch(&"a");

        let entries: Vec<_> = buckets.iter_bucket_entries(1).collect();
        assert_eq!(entries.len(), 1);
        let (id, meta) = entries[0];
        assert_eq!(meta.key, &"b");
        assert_eq!(meta.freq, 1);
        assert_eq!(meta.last_epoch, 0);
        assert_eq!(buckets.entries.get(id).unwrap().freq, 1);
    }

    #[test]
    fn frequency_buckets_iter_entries_meta() {
        let mut buckets = FrequencyBuckets::new();
        buckets.insert("a");
        buckets.touch(&"a");
        buckets.insert("b");

        let mut entries: Vec<_> = buckets.iter_entries().collect();
        entries.sort_by_key(|(id, _)| id.index());
        assert_eq!(entries.len(), 2);
        let metas: Vec<_> = entries
            .into_iter()
            .map(|(_, meta)| (meta.key, meta.freq))
            .collect();
        assert!(metas.contains(&(&"a", 2)));
        assert!(metas.contains(&(&"b", 1)));
    }

    #[test]
    fn frequency_buckets_debug_snapshot() {
        let mut buckets = FrequencyBuckets::new();
        buckets.insert("a");
        buckets.insert("b");
        buckets.touch(&"a");
        let snapshot = buckets.debug_snapshot();
        assert_eq!(snapshot.min_freq, Some(1));
        assert_eq!(snapshot.entries_len, 2);
        assert_eq!(snapshot.index_len, 2);
        assert_eq!(snapshot.buckets.len(), 2);
        assert_eq!(snapshot.epoch, 0);
        assert_eq!(snapshot.entry_epochs.len(), 2);
        assert!(snapshot.entry_epochs.iter().all(|(_, epoch)| *epoch == 0));
        assert_eq!(snapshot.bucket_entries.len(), 2);
        let first_bucket = &snapshot.bucket_entries[0].1;
        assert_eq!(first_bucket.len(), 1);
        assert_eq!(first_bucket[0].freq, 1);
        assert!(matches!(first_bucket[0].key, "a" | "b"));
    }

    #[test]
    fn frequency_buckets_borrowed_key_lookups() {
        let mut buckets = FrequencyBuckets::new();
        buckets.insert("alpha".to_string());
        buckets.touch(&"alpha".to_string());

        assert!(buckets.contains_borrowed("alpha"));
        assert_eq!(buckets.frequency_borrowed("alpha"), Some(2));
    }

    #[test]
    fn frequency_buckets_batch_ops() {
        let mut buckets = FrequencyBuckets::new();
        assert_eq!(buckets.insert_batch(["a", "b", "c"]), 3);
        assert_eq!(buckets.touch_batch(["a", "b", "z"]), 2);
        assert_eq!(buckets.remove_batch(["b", "c", "z"]), 2);
        assert_eq!(buckets.len(), 1);
    }

    #[test]
    fn frequency_buckets_touch_capped() {
        let mut buckets = FrequencyBuckets::new();
        buckets.insert("a");
        assert_eq!(buckets.touch_capped(&"a", 2), Some(2));
        assert_eq!(buckets.touch_capped(&"a", 2), Some(2));
        assert_eq!(buckets.frequency(&"a"), Some(2));
    }

    #[test]
    fn frequency_buckets_decay_halve() {
        let mut buckets = FrequencyBuckets::new();
        buckets.insert("a");
        buckets.insert("b");
        buckets.touch(&"a");
        buckets.touch(&"a");
        assert_eq!(buckets.frequency(&"a"), Some(3));
        buckets.decay_halve();
        assert_eq!(buckets.frequency(&"a"), Some(1));
        assert_eq!(buckets.min_freq(), Some(1));
    }

    #[test]
    fn frequency_buckets_rebase_min_freq() {
        let mut buckets = FrequencyBuckets::new();
        buckets.insert("a");
        buckets.insert("b");
        buckets.touch(&"a");
        buckets.touch(&"a");
        assert_eq!(buckets.frequency(&"a"), Some(3));
        buckets.remove(&"b");
        assert_eq!(buckets.min_freq(), Some(3));
        buckets.rebase_min_freq();
        assert_eq!(buckets.frequency(&"a"), Some(1));
        assert_eq!(buckets.min_freq(), Some(1));
    }

    #[cfg(feature = "concurrency")]
    #[test]
    fn sharded_frequency_buckets_basic_ops() {
        let buckets = ShardedFrequencyBuckets::new(2);
        assert!(buckets.insert("a"));
        assert!(buckets.insert("b"));
        assert!(buckets.contains(&"a"));
        assert_eq!(buckets.touch(&"a"), Some(2));
        assert_eq!(buckets.frequency(&"a"), Some(2));
        let peeked = buckets.peek_min();
        assert!(matches!(peeked, Some(("a", 2)) | Some(("b", 1))));
        let popped = buckets.pop_min();
        assert!(matches!(popped, Some(("a", 2)) | Some(("b", 1))));
        assert_eq!(buckets.len(), 1);
    }

    #[cfg(feature = "concurrency")]
    #[test]
    fn sharded_frequency_buckets_shard_for_key() {
        let buckets: ShardedFrequencyBuckets<&str> = ShardedFrequencyBuckets::new(4);
        let shard = buckets.shard_for_key(&"alpha");
        assert!(shard < buckets.shard_count());
    }

    #[cfg(feature = "concurrency")]
    #[test]
    fn sharded_frequency_buckets_with_seed() {
        let buckets: ShardedFrequencyBuckets<&str> =
            ShardedFrequencyBuckets::with_shards_seed(4, 0, 99);
        let shard = buckets.shard_for_key(&"alpha");
        assert!(shard < buckets.shard_count());
    }

    #[cfg(feature = "concurrency")]
    #[test]
    fn sharded_frequency_buckets_iter_entries() {
        let buckets = ShardedFrequencyBuckets::new(2);
        assert!(buckets.insert("a"));
        assert!(buckets.insert("b"));
        assert_eq!(buckets.touch(&"a"), Some(2));

        let mut entries = buckets.iter_entries();
        entries.sort_by_key(|(_, meta)| meta.key);
        let metas: Vec<_> = entries
            .into_iter()
            .map(|(_, meta)| (meta.key, meta.freq, meta.last_epoch))
            .collect();
        assert!(metas.contains(&("a", 2, 0)));
        assert!(metas.contains(&("b", 1, 0)));

        let mut bucket_entries = buckets.iter_bucket_entries(1);
        bucket_entries.sort_by_key(|(_, meta)| meta.key);
        assert_eq!(bucket_entries.len(), 1);
        assert_eq!(bucket_entries[0].1.key, "b");

        let mut shard_entries = buckets.iter_entries_with_shard();
        shard_entries.sort_by_key(|(_, _, meta)| meta.key);
        assert_eq!(shard_entries.len(), 2);
        assert!(
            shard_entries
                .iter()
                .all(|(idx, _, _)| *idx < buckets.shard_count())
        );

        let mut shard_bucket_entries = buckets.iter_bucket_entries_with_shard(1);
        shard_bucket_entries.sort_by_key(|(_, _, meta)| meta.key);
        assert_eq!(shard_bucket_entries.len(), 1);
        assert_eq!(shard_bucket_entries[0].2.key, "b");
    }

    #[test]
    fn frequency_buckets_handle_basic_ops() {
        let mut buckets = FrequencyBucketsHandle::new();
        assert!(buckets.insert(1u64));
        assert!(buckets.insert(2u64));
        assert!(buckets.contains(&1));
        assert_eq!(buckets.touch(&1), Some(2));
        assert_eq!(buckets.frequency(&1), Some(2));
        let peeked = buckets.peek_min();
        assert!(matches!(peeked, Some((1, 2)) | Some((2, 1))));
        let popped = buckets.pop_min();
        assert!(matches!(popped, Some((1, 2)) | Some((2, 1))));
        assert_eq!(buckets.len(), 1);
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    // =============================================================================
    // Property Tests - Core Invariants
    // =============================================================================

    proptest! {
        /// Property: Invariants hold after any sequence of operations
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_invariants_always_hold(
            ops in prop::collection::vec((0u8..4, any::<u32>()), 0..100)
        ) {
            let mut buckets: FrequencyBuckets<u32> = FrequencyBuckets::new();

            for (op, key) in ops {
                match op % 4 {
                    0 => { buckets.insert(key); }
                    1 => { buckets.touch(&key); }
                    2 => { buckets.remove(&key); }
                    3 => { buckets.pop_min(); }
                    _ => unreachable!(),
                }

                buckets.debug_validate_invariants();
            }
        }

        /// Property: len() equals number of unique keys inserted minus removed
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_len_is_accurate(
            ops in prop::collection::vec((0u8..3, 0u32..20), 0..50)
        ) {
            let mut buckets: FrequencyBuckets<u32> = FrequencyBuckets::new();
            let mut expected_keys = std::collections::HashSet::new();

            for (op, key) in ops {
                match op % 3 {
                    0 => {
                        buckets.insert(key);
                        expected_keys.insert(key);
                    }
                    1 => {
                        buckets.remove(&key);
                        expected_keys.remove(&key);
                    }
                    2 => {
                        if let Some((evicted_key, _)) = buckets.pop_min() {
                            expected_keys.remove(&evicted_key);
                        }
                    }
                    _ => unreachable!(),
                }

                prop_assert_eq!(buckets.len(), expected_keys.len());
            }
        }

        /// Property: Empty state is consistent
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_empty_state_consistent(
            keys in prop::collection::vec(any::<u32>(), 0..20)
        ) {
            let mut buckets: FrequencyBuckets<u32> = FrequencyBuckets::new();

            for key in keys {
                buckets.insert(key);
            }

            while !buckets.is_empty() {
                buckets.pop_min();
            }

            prop_assert!(buckets.is_empty());
            prop_assert_eq!(buckets.len(), 0);
            prop_assert_eq!(buckets.min_freq(), None);
            prop_assert_eq!(buckets.peek_min(), None);
            prop_assert_eq!(buckets.pop_min(), None);
        }
    }

    // =============================================================================
    // Property Tests - Frequency Tracking
    // =============================================================================

    proptest! {
        /// Property: Frequencies start at 1 and increment by 1
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_frequency_starts_at_one_and_increments(
            key in any::<u32>(),
            touch_count in 0usize..20
        ) {
            let mut buckets: FrequencyBuckets<u32> = FrequencyBuckets::new();

            buckets.insert(key);
            prop_assert_eq!(buckets.frequency(&key), Some(1));

            for i in 0..touch_count {
                let new_freq = buckets.touch(&key).unwrap();
                prop_assert_eq!(new_freq, 2 + i as u64);
                prop_assert_eq!(buckets.frequency(&key), Some(2 + i as u64));
            }
        }

        /// Property: touch returns None for missing keys
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_touch_missing_returns_none(
            present_key in any::<u32>(),
            missing_key in any::<u32>()
        ) {
            prop_assume!(present_key != missing_key);
            let mut buckets: FrequencyBuckets<u32> = FrequencyBuckets::new();
            buckets.insert(present_key);

            prop_assert_eq!(buckets.touch(&missing_key), None);
            prop_assert!(!buckets.contains(&missing_key));
        }

        /// Property: touch_capped respects max frequency
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_touch_capped_respects_max(
            key in any::<u32>(),
            max_freq in 2u64..10,
            touch_count in 0usize..20
        ) {
            let mut buckets: FrequencyBuckets<u32> = FrequencyBuckets::new();
            buckets.insert(key);

            for _ in 0..touch_count {
                buckets.touch_capped(&key, max_freq);
            }

            let final_freq = buckets.frequency(&key).unwrap();
            prop_assert!(final_freq <= max_freq);
        }
    }

    // =============================================================================
    // Property Tests - min_freq Tracking
    // =============================================================================

    proptest! {
        /// Property: min_freq is always the minimum frequency present
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_min_freq_is_accurate(
            ops in prop::collection::vec((0u8..2, 0u32..10), 1..30)
        ) {
            let mut buckets: FrequencyBuckets<u32> = FrequencyBuckets::new();

            for (op, key) in ops {
                match op % 2 {
                    0 => { buckets.insert(key); }
                    1 => { buckets.touch(&key); }
                    _ => unreachable!(),
                }

                // Compute actual minimum frequency
                let mut actual_min: Option<u64> = None;
                for (_, meta) in buckets.iter_entries() {
                    actual_min = match actual_min {
                        None => Some(meta.freq),
                        Some(min) => Some(min.min(meta.freq)),
                    };
                }

                prop_assert_eq!(buckets.min_freq(), actual_min);
            }
        }

        /// Property: min_freq updates correctly after removal
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_min_freq_updates_after_removal(
            keys in prop::collection::vec(0u32..10, 2..10)
        ) {
            let mut buckets: FrequencyBuckets<u32> = FrequencyBuckets::new();
            let mut unique_keys = Vec::new();
            let mut seen = std::collections::HashSet::new();
            for key in keys {
                if seen.insert(key) {
                    unique_keys.push(key);
                }
            }

            // Insert all keys
            for &key in &unique_keys {
                buckets.insert(key);
            }

            // Touch first key multiple times
            if let Some(&first_key) = unique_keys.first() {
                for _ in 0..5 {
                    buckets.touch(&first_key);
                }
            }

            // min_freq should be 1 if we have other keys, else 6 for the touched key
            let expected_min = if unique_keys.len() > 1 { Some(1) } else { Some(6) };
            prop_assert_eq!(buckets.min_freq(), expected_min);

            // Remove all keys at freq=1
            for &key in &unique_keys {
                if buckets.frequency(&key) == Some(1) {
                    buckets.remove(&key);
                }
            }

            // min_freq should now be 6 (first key) or None if empty
            if !buckets.is_empty() {
                prop_assert!(buckets.min_freq().unwrap() > 1);
            }
        }
    }

    // =============================================================================
    // Property Tests - FIFO Ordering
    // =============================================================================

    proptest! {
        /// Property: FIFO order within same frequency bucket
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_fifo_order_same_frequency(
            keys in prop::collection::vec(0u32..50, 2..20)
        ) {
            let mut buckets: FrequencyBuckets<u32> = FrequencyBuckets::new();
            let mut unique_keys = Vec::new();
            let mut seen = std::collections::HashSet::new();
            for key in keys {
                if seen.insert(key) {
                    unique_keys.push(key);
                }
            }

            // Insert keys in order
            for &key in &unique_keys {
                buckets.insert(key);
            }

            // All keys at freq=1, should pop in FIFO order
            for expected_key in unique_keys {
                let (popped_key, freq) = buckets.pop_min().unwrap();
                prop_assert_eq!(popped_key, expected_key);
                prop_assert_eq!(freq, 1);
            }

            prop_assert!(buckets.is_empty());
        }
    }

    // =============================================================================
    // Property Tests - Eviction (pop_min)
    // =============================================================================

    proptest! {
        /// Property: peek_min and pop_min return the same key/freq
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_peek_pop_consistency(
            keys in prop::collection::vec(any::<u32>(), 1..20),
            touch_ops in prop::collection::vec((any::<u32>(), 0usize..5), 0..20)
        ) {
            let mut buckets: FrequencyBuckets<u32> = FrequencyBuckets::new();

            for key in keys {
                buckets.insert(key);
            }

            for (key, count) in touch_ops {
                for _ in 0..count {
                    buckets.touch(&key);
                }
            }

            while !buckets.is_empty() {
                let peeked = buckets.peek_min().map(|(k, f)| (*k, f));
                let popped = buckets.pop_min();

                prop_assert!(peeked.is_some());
                prop_assert!(popped.is_some());

                let (peek_key, peek_freq) = peeked.unwrap();
                let (pop_key, pop_freq) = popped.unwrap();

                prop_assert_eq!(peek_key, pop_key);
                prop_assert_eq!(peek_freq, pop_freq);
            }
        }

        /// Property: pop_min returns lowest frequency
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_pop_min_returns_lowest_frequency(
            keys in prop::collection::vec(0u32..10, 2..10)
        ) {
            let mut buckets: FrequencyBuckets<u32> = FrequencyBuckets::new();

            // Insert keys
            for &key in &keys {
                buckets.insert(key);
            }

            // Touch some keys
            if let Some(&first_key) = keys.first() {
                for _ in 0..3 {
                    buckets.touch(&first_key);
                }
            }

            // Pop should return a key with the minimum frequency
            if let Some((_, popped_freq)) = buckets.pop_min() {
                // All remaining keys should have freq >= popped_freq
                for (_, meta) in buckets.iter_entries() {
                    prop_assert!(meta.freq >= popped_freq);
                }
            }
        }
    }

    // =============================================================================
    // Property Tests - Remove Operations
    // =============================================================================

    proptest! {
        /// Property: remove returns correct frequency and removes key
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_remove_returns_freq_and_removes(
            key in any::<u32>(),
            touch_count in 0usize..10
        ) {
            let mut buckets: FrequencyBuckets<u32> = FrequencyBuckets::new();
            buckets.insert(key);

            for _ in 0..touch_count {
                buckets.touch(&key);
            }

            let expected_freq = 1 + touch_count as u64;
            let removed_freq = buckets.remove(&key).unwrap();

            prop_assert_eq!(removed_freq, expected_freq);
            prop_assert!(!buckets.contains(&key));
            prop_assert_eq!(buckets.frequency(&key), None);
        }

        /// Property: remove missing key returns None
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_remove_missing_returns_none(
            present_key in any::<u32>(),
            missing_key in any::<u32>()
        ) {
            prop_assume!(present_key != missing_key);
            let mut buckets: FrequencyBuckets<u32> = FrequencyBuckets::new();
            buckets.insert(present_key);

            prop_assert_eq!(buckets.remove(&missing_key), None);
            prop_assert!(!buckets.contains(&missing_key));
        }
    }

    // =============================================================================
    // Property Tests - Clear Operations
    // =============================================================================

    proptest! {
        /// Property: clear resets to empty state
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_clear_resets_state(
            keys in prop::collection::vec(any::<u32>(), 1..30)
        ) {
            let mut buckets: FrequencyBuckets<u32> = FrequencyBuckets::new();

            for key in keys {
                buckets.insert(key);
                buckets.touch(&key);
            }

            buckets.clear();

            prop_assert!(buckets.is_empty());
            prop_assert_eq!(buckets.len(), 0);
            prop_assert_eq!(buckets.min_freq(), None);
            prop_assert_eq!(buckets.peek_min(), None);
            prop_assert_eq!(buckets.pop_min(), None);
            buckets.debug_validate_invariants();
        }

        /// Property: clear_shrink behaves like clear
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_clear_shrink_same_as_clear(
            keys in prop::collection::vec(any::<u32>(), 1..30)
        ) {
            let mut buckets1: FrequencyBuckets<u32> = FrequencyBuckets::new();
            let mut buckets2: FrequencyBuckets<u32> = FrequencyBuckets::new();

            for &key in &keys {
                buckets1.insert(key);
                buckets2.insert(key);
            }

            buckets1.clear();
            buckets2.clear_shrink();

            prop_assert_eq!(buckets1.len(), buckets2.len());
            prop_assert_eq!(buckets1.is_empty(), buckets2.is_empty());
            prop_assert_eq!(buckets1.min_freq(), buckets2.min_freq());
        }

        /// Property: usable after clear
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_usable_after_clear(
            keys1 in prop::collection::vec(any::<u32>(), 1..20),
            keys2 in prop::collection::vec(any::<u32>(), 1..20)
        ) {
            let mut buckets: FrequencyBuckets<u32> = FrequencyBuckets::new();

            for key in keys1 {
                buckets.insert(key);
            }

            buckets.clear();

            for key in &keys2 {
                buckets.insert(*key);
            }

            let unique_keys2: std::collections::HashSet<_> = keys2.into_iter().collect();
            prop_assert_eq!(buckets.len(), unique_keys2.len());
        }
    }

    // =============================================================================
    // Property Tests - Decay Operations
    // =============================================================================

    proptest! {
        /// Property: decay_halve reduces frequencies correctly
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_decay_halve_reduces_frequencies(
            keys in prop::collection::vec(0u32..10, 1..10)
        ) {
            let mut buckets: FrequencyBuckets<u32> = FrequencyBuckets::new();

            // Insert and build up frequencies
            for &key in &keys {
                buckets.insert(key);
                for _ in 0..5 {
                    buckets.touch(&key);
                }
            }

            // Record frequencies before decay
            let mut before: std::collections::HashMap<u32, u64> = std::collections::HashMap::new();
            for (_, meta) in buckets.iter_entries() {
                before.insert(*meta.key, meta.freq);
            }

            buckets.decay_halve();

            // Check frequencies after decay
            for (key, old_freq) in before {
                let new_freq = buckets.frequency(&key).unwrap();
                let expected = (old_freq / 2).max(1);
                prop_assert_eq!(new_freq, expected);
            }

            buckets.debug_validate_invariants();
        }

        /// Property: rebase_min_freq shifts frequencies correctly
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_rebase_min_freq_shifts_correctly(
            keys in prop::collection::vec(0u32..5, 2..8)
        ) {
            let mut buckets: FrequencyBuckets<u32> = FrequencyBuckets::new();

            // Insert keys with varying frequencies
            for (i, &key) in keys.iter().enumerate() {
                buckets.insert(key);
                for _ in 0..i {
                    buckets.touch(&key);
                }
            }

            let min_before = buckets.min_freq();
            if min_before.unwrap_or(0) <= 1 {
                return Ok(());
            }

            let delta = min_before.unwrap() - 1;

            // Record frequencies before rebase
            let mut before: std::collections::HashMap<u32, u64> = std::collections::HashMap::new();
            for (_, meta) in buckets.iter_entries() {
                before.insert(*meta.key, meta.freq);
            }

            buckets.rebase_min_freq();

            // Check frequencies after rebase
            for (key, old_freq) in before {
                let new_freq = buckets.frequency(&key).unwrap();
                let expected = old_freq.saturating_sub(delta).max(1);
                prop_assert_eq!(new_freq, expected);
            }

            prop_assert_eq!(buckets.min_freq(), Some(1));
            buckets.debug_validate_invariants();
        }
    }

    // =============================================================================
    // Property Tests - Batch Operations
    // =============================================================================

    proptest! {
        /// Property: insert_batch inserts correct number of unique keys
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_insert_batch_unique_count(
            keys in prop::collection::vec(any::<u32>(), 0..30)
        ) {
            let mut buckets: FrequencyBuckets<u32> = FrequencyBuckets::new();
            let unique_keys: std::collections::HashSet<_> = keys.iter().copied().collect();

            let inserted = buckets.insert_batch(keys);

            prop_assert_eq!(inserted, unique_keys.len());
            prop_assert_eq!(buckets.len(), unique_keys.len());
        }

        /// Property: touch_batch touches only present keys
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_touch_batch_only_present(
            present_keys in prop::collection::vec(0u32..10, 1..10),
            touch_keys in prop::collection::vec(0u32..20, 0..10)
        ) {
            let mut buckets: FrequencyBuckets<u32> = FrequencyBuckets::new();

            buckets.insert_batch(present_keys.clone());
            let touched = buckets.touch_batch(touch_keys.clone());

            let present_set: std::collections::HashSet<_> = present_keys.into_iter().collect();
            let intersection_count = touch_keys
                .iter()
                .filter(|key| present_set.contains(key))
                .count();

            prop_assert_eq!(touched, intersection_count);
        }
    }
}
