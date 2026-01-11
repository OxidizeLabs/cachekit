//! Frequency buckets for O(1) LFU tracking.
//!
//! Maintains:
//! - `HashMap<K, SlotId>` index for key lookup
//! - `SlotArena<Entry<K>>` for per-key frequency + links
//! - `HashMap<u64, Bucket>` for per-frequency FIFO lists
//! - `min_freq` pointer for O(1) eviction
//!
//! ## Architecture
//!
//! ```text
//!   index (key -> SlotId)          entries (SlotArena<Entry<K>>)
//!   ┌─────────┬─────────┐          ┌───────────────────────────────┐
//!   │  key A  │  id_7   │          │ Entry { key: A, freq: 2, ... }│
//!   │  key B  │  id_3   │  ──────► │ Entry { key: B, freq: 1, ... }│
//!   └─────────┴─────────┘          └───────────────────────────────┘
//!
//!   buckets (freq -> list of SlotId)
//!   freq=1: head ─► [id_3] ◄──► [id_9] ◄── tail  (FIFO within bucket)
//!   freq=2: head ─► [id_7] ◄── tail
//!   min_freq → 1
//! ```
//!
//! ## Eviction Flow
//!
//! ```text
//!   pop_min():
//!     1. Use min_freq to find lowest bucket
//!     2. Pop tail SlotId (oldest in that bucket)
//!     3. Remove entry + update min_freq if bucket empties
//! ```
//!
//! ## Performance
//! - `insert` / `touch` / `remove` / `pop_min`: O(1) average
//! - FIFO tie-breaking within a frequency bucket
//! - `decay_halve` / `rebase_min_freq`: O(n) rebuild of bucket structure
//!
//! ## Handle-Based Usage
//!
//! For large keys, consider interning keys in a higher layer and using
//! `FrequencyBucketsHandle<Handle>` where `Handle: Copy + Eq + Hash`.
//! This stores the handle (not the full key) in buckets and index maps.
//!
//! `debug_validate_invariants()` is available in debug/test builds.
use std::collections::HashMap;
use std::hash::Hash;
use std::hash::Hasher;

use crate::ds::slot_arena::{SlotArena, SlotId};

#[derive(Debug)]
struct Entry<K> {
    key: K,
    freq: u64,
    prev: Option<SlotId>,
    next: Option<SlotId>,
}

#[derive(Debug, Default)]
struct Bucket {
    head: Option<SlotId>,
    tail: Option<SlotId>,
    prev: Option<u64>,
    next: Option<u64>,
}

#[derive(Debug)]
/// O(1) LFU metadata tracker with FIFO tie-breaking within a frequency.
pub struct FrequencyBuckets<K> {
    entries: SlotArena<Entry<K>>,
    index: HashMap<K, SlotId>,
    buckets: HashMap<u64, Bucket>,
    min_freq: u64,
}

impl<K> FrequencyBuckets<K>
where
    K: Eq + Hash + Clone,
{
    /// Creates an empty tracker.
    pub fn new() -> Self {
        Self {
            entries: SlotArena::new(),
            index: HashMap::new(),
            buckets: HashMap::new(),
            min_freq: 0,
        }
    }

    /// Returns the number of tracked keys.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if there are no tracked keys.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns `true` if `key` is present.
    pub fn contains(&self, key: &K) -> bool {
        self.index.contains_key(key)
    }

    /// Returns the current frequency for `key`, if present.
    pub fn frequency(&self, key: &K) -> Option<u64> {
        let id = *self.index.get(key)?;
        self.entries.get(id).map(|entry| entry.freq)
    }

    /// Returns the minimum frequency currently present.
    pub fn min_freq(&self) -> Option<u64> {
        if self.min_freq == 0 {
            None
        } else {
            Some(self.min_freq)
        }
    }

    /// Peeks the eviction candidate `(key, freq)` (tail of the min-frequency bucket).
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
    pub fn peek_min_id(&self) -> Option<SlotId> {
        if self.min_freq == 0 {
            return None;
        }
        let bucket = self.buckets.get(&self.min_freq)?;
        bucket.tail
    }

    /// Peeks the key for the eviction candidate (tail of the min-frequency bucket).
    pub fn peek_min_key(&self) -> Option<&K> {
        let id = self.peek_min_id()?;
        self.entries.get(id).map(|entry| &entry.key)
    }

    /// Returns an iterator of SlotIds for a given frequency, from head to tail.
    pub fn iter_bucket_ids(&self, freq: u64) -> FrequencyBucketIdIter<'_, K> {
        let head = self.buckets.get(&freq).and_then(|bucket| bucket.head);
        FrequencyBucketIdIter {
            buckets: self,
            current: head,
        }
    }

    /// Inserts a new key with frequency 1.
    ///
    /// Returns `false` if the key already exists.
    pub fn insert(&mut self, key: K) -> bool {
        if self.index.contains_key(&key) {
            return false;
        }

        let id = self.entries.insert(Entry {
            key: key.clone(),
            freq: 1,
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

    /// Increments frequency for `key` and returns the new frequency.
    ///
    /// Returns `None` if `key` is missing. Within each frequency bucket,
    /// the key is treated as MRU by being pushed to the front.
    pub fn touch(&mut self, key: &K) -> Option<u64> {
        let id = *self.index.get(key)?;
        let current_freq = self.entries.get(id)?.freq;
        if current_freq == u64::MAX {
            self.list_remove(current_freq, id)?;
            self.list_push_front(current_freq, id);
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
        }
        self.list_push_front(next_freq, id);
        if self.min_freq == 0 || next_freq < self.min_freq {
            self.min_freq = next_freq;
        }

        Some(next_freq)
    }

    /// Increments frequency for `key`, clamping at `max_freq`.
    ///
    /// If the key is already at `max_freq`, it is moved to the front of its
    /// bucket and the frequency is unchanged.
    pub fn touch_capped(&mut self, key: &K, max_freq: u64) -> Option<u64> {
        let max_freq = max_freq.max(1);
        let id = *self.index.get(key)?;
        let current_freq = self.entries.get(id)?.freq;
        if current_freq >= max_freq {
            self.list_remove(current_freq, id)?;
            self.list_push_front(current_freq, id);
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
    pub fn decay_halve(&mut self) {
        if self.is_empty() {
            return;
        }
        self.rebuild_with(|freq| (freq / 2).max(1));
    }

    /// Rebases frequencies so the current minimum becomes 1.
    ///
    /// This is an O(n) rebuild and will reorder tie-breaks within buckets.
    pub fn rebase_min_freq(&mut self) {
        if self.min_freq <= 1 {
            return;
        }
        let delta = self.min_freq - 1;
        self.rebuild_with(|freq| freq.saturating_sub(delta).max(1));
    }

    /// Removes `key` from tracking and returns its previous frequency.
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

    /// Removes and returns the eviction candidate `(key, freq)`.
    ///
    /// Eviction is O(1) using `min_freq` and the tail of that bucket.
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
    pub fn clear(&mut self) {
        self.entries.clear();
        self.index.clear();
        self.buckets.clear();
        self.min_freq = 0;
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
    pub fn debug_snapshot(&self) -> FrequencyBucketsSnapshot {
        let mut buckets: Vec<(u64, Vec<SlotId>)> = self
            .buckets
            .keys()
            .copied()
            .map(|freq| (freq, self.iter_bucket_ids(freq).collect()))
            .collect();
        buckets.sort_by_key(|(freq, _)| *freq);
        FrequencyBucketsSnapshot {
            min_freq: self.min_freq(),
            entries_len: self.entries.len(),
            index_len: self.index.len(),
            buckets,
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

        if let Some(prev) = prev
            && let Some(prev_bucket) = self.buckets.get_mut(&prev)
        {
            prev_bucket.next = Some(freq);
        }
        if let Some(next) = next
            && let Some(next_bucket) = self.buckets.get_mut(&next)
        {
            next_bucket.prev = Some(freq);
        }
    }

    fn remove_bucket(&mut self, freq: u64, prev: Option<u64>, next: Option<u64>) {
        if let Some(prev) = prev
            && let Some(prev_bucket) = self.buckets.get_mut(&prev)
        {
            prev_bucket.next = next;
        }
        if let Some(next) = next
            && let Some(next_bucket) = self.buckets.get_mut(&next)
        {
            next_bucket.prev = prev;
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
pub struct FrequencyBucketsSnapshot {
    pub min_freq: Option<u64>,
    pub entries_len: usize,
    pub index_len: usize,
    pub buckets: Vec<(u64, Vec<SlotId>)>,
}

#[derive(Debug)]
/// Sharded frequency buckets for reduced contention.
pub struct ShardedFrequencyBuckets<K> {
    shards: Vec<parking_lot::RwLock<FrequencyBuckets<K>>>,
}

impl<K> ShardedFrequencyBuckets<K>
where
    K: Eq + Hash + Clone,
{
    /// Creates a sharded tracker with `shards` shards.
    pub fn new(shards: usize) -> Self {
        let shards = shards.max(1);
        let mut vec = Vec::with_capacity(shards);
        for _ in 0..shards {
            vec.push(parking_lot::RwLock::new(FrequencyBuckets::new()));
        }
        Self { shards: vec }
    }

    /// Returns the number of shards.
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    fn shard_for(&self, key: &K) -> usize {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        (hasher.finish() as usize) % self.shards.len()
    }

    /// Inserts a key into its shard.
    pub fn insert(&self, key: K) -> bool {
        let shard = self.shard_for(&key);
        let mut buckets = self.shards[shard].write();
        buckets.insert(key)
    }

    /// Touches a key in its shard.
    pub fn touch(&self, key: &K) -> Option<u64> {
        let shard = self.shard_for(key);
        let mut buckets = self.shards[shard].write();
        buckets.touch(key)
    }

    /// Removes a key from its shard.
    pub fn remove(&self, key: &K) -> Option<u64> {
        let shard = self.shard_for(key);
        let mut buckets = self.shards[shard].write();
        buckets.remove(key)
    }

    /// Returns the frequency for a key in its shard.
    pub fn frequency(&self, key: &K) -> Option<u64> {
        let shard = self.shard_for(key);
        let buckets = self.shards[shard].read();
        buckets.frequency(key)
    }

    /// Returns `true` if the key exists in its shard.
    pub fn contains(&self, key: &K) -> bool {
        let shard = self.shard_for(key);
        let buckets = self.shards[shard].read();
        buckets.contains(key)
    }

    /// Returns the total number of keys across all shards.
    pub fn len(&self) -> usize {
        self.shards.iter().map(|b| b.read().len()).sum()
    }

    /// Returns `true` if all shards are empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clears all shards.
    pub fn clear(&self) {
        for shard in &self.shards {
            shard.write().clear();
        }
    }

    /// Peeks the global min across shards by cloning the candidate.
    pub fn peek_min(&self) -> Option<(K, u64)> {
        let mut best: Option<(usize, u64)> = None;
        for (idx, shard) in self.shards.iter().enumerate() {
            let buckets = shard.read();
            if let Some(freq) = buckets.min_freq()
                && best.is_none_or(|(_, best_freq)| freq < best_freq)
            {
                best = Some((idx, freq));
            }
        }
        let (idx, _) = best?;
        let buckets = self.shards[idx].read();
        buckets.peek_min().map(|(key, freq)| (key.clone(), freq))
    }

    /// Pops the global min across shards (scan by min_freq).
    pub fn pop_min(&self) -> Option<(K, u64)> {
        let mut best: Option<(usize, u64)> = None;
        for (idx, shard) in self.shards.iter().enumerate() {
            let buckets = shard.read();
            if let Some(freq) = buckets.min_freq()
                && best.is_none_or(|(_, best_freq)| freq < best_freq)
            {
                best = Some((idx, freq));
            }
        }
        let (idx, _) = best?;
        let mut buckets = self.shards[idx].write();
        buckets.pop_min()
    }
}

#[derive(Debug)]
/// Frequency buckets keyed by a compact handle (for interned keys).
pub struct FrequencyBucketsHandle<H> {
    inner: FrequencyBuckets<H>,
}

impl<H> FrequencyBucketsHandle<H>
where
    H: Eq + Hash + Copy,
{
    /// Creates an empty tracker.
    pub fn new() -> Self {
        Self {
            inner: FrequencyBuckets::new(),
        }
    }

    /// Returns the number of tracked handles.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if there are no tracked handles.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns `true` if `handle` is present.
    pub fn contains(&self, handle: &H) -> bool {
        self.inner.contains(handle)
    }

    /// Returns the current frequency for `handle`, if present.
    pub fn frequency(&self, handle: &H) -> Option<u64> {
        self.inner.frequency(handle)
    }

    /// Returns the minimum frequency currently present.
    pub fn min_freq(&self) -> Option<u64> {
        self.inner.min_freq()
    }

    /// Peeks the eviction candidate `(handle, freq)`.
    pub fn peek_min(&self) -> Option<(H, u64)> {
        self.inner.peek_min().map(|(handle, freq)| (*handle, freq))
    }

    /// Peeks the eviction candidate by reference `(handle, freq)`.
    pub fn peek_min_ref(&self) -> Option<(&H, u64)> {
        self.inner.peek_min()
    }

    /// Inserts a new handle with frequency 1.
    pub fn insert(&mut self, handle: H) -> bool {
        self.inner.insert(handle)
    }

    /// Increments frequency for `handle` and returns the new frequency.
    pub fn touch(&mut self, handle: &H) -> Option<u64> {
        self.inner.touch(handle)
    }

    /// Increments frequency for `handle`, clamping at `max_freq`.
    pub fn touch_capped(&mut self, handle: &H, max_freq: u64) -> Option<u64> {
        self.inner.touch_capped(handle, max_freq)
    }

    /// Removes `handle` from tracking and returns its previous frequency.
    pub fn remove(&mut self, handle: &H) -> Option<u64> {
        self.inner.remove(handle)
    }

    /// Removes and returns the eviction candidate `(handle, freq)`.
    pub fn pop_min(&mut self) -> Option<(H, u64)> {
        self.inner.pop_min()
    }

    /// Halves all frequencies (rounding down), clamping at 1.
    pub fn decay_halve(&mut self) {
        self.inner.decay_halve();
    }

    /// Rebases frequencies so the current minimum becomes 1.
    pub fn rebase_min_freq(&mut self) {
        self.inner.rebase_min_freq();
    }

    /// Clears all state.
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    #[cfg(any(test, debug_assertions))]
    /// Returns a debug snapshot of bucket chains.
    pub fn debug_snapshot(&self) -> FrequencyBucketsSnapshot {
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
