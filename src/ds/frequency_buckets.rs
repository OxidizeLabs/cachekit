use std::collections::HashMap;
use std::hash::Hash;

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
    pub fn new() -> Self {
        Self {
            entries: SlotArena::new(),
            index: HashMap::new(),
            buckets: HashMap::new(),
            min_freq: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn contains(&self, key: &K) -> bool {
        self.index.contains_key(key)
    }

    pub fn frequency(&self, key: &K) -> Option<u64> {
        let id = *self.index.get(key)?;
        self.entries.get(id).map(|entry| entry.freq)
    }

    pub fn min_freq(&self) -> Option<u64> {
        if self.min_freq == 0 {
            None
        } else {
            Some(self.min_freq)
        }
    }

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

    pub fn clear(&mut self) {
        self.entries.clear();
        self.index.clear();
        self.buckets.clear();
        self.min_freq = 0;
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
}
