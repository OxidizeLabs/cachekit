//! Clock-sweep ring for second-chance eviction.
//!
//! Uses a fixed-size slot array and a hand pointer to evict the first
//! unreferenced entry encountered. Accesses set a referenced bit that
//! grants a second chance before eviction.
//!
//! ## Architecture
//!
//! ```text
//!   ┌──────────────────────────────────────────────────────────────────────┐
//!   │                         ClockRing<K, V>                               │
//!   │                                                                       │
//!   │   slots: Vec<Option<Entry<K,V>>>                                      │
//!   │   hand ──────────────────────────────────────────────┐                │
//!   │                                                      │                │
//!   │   index: HashMap<K, usize> (key -> slot index)        │                │
//!   │   ┌─────────┬─────────┐                               ▼                │
//!   │   │  key A  │   0     │   slot[0] = Entry { ref:1 }  [A]               │
//!   │   │  key B  │   1     │   slot[1] = Entry { ref:0 }  [B]               │
//!   │   │  key C  │   2     │   slot[2] = Entry { ref:1 }  [C]               │
//!   │   └─────────┴─────────┘   slot[3] = None              [ ]               │
//!   │                                                                       │
//!   │   Eviction scan (hand moves forward):                                 │
//!   │   [A ref=1] -> clear ref, advance                                     │
//!   │   [B ref=0] -> evict B, insert new entry here                         │
//!   └──────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Eviction Flow
//!
//! ```text
//!   insert(key, value)
//!        │
//!        ▼
//!   ┌──────────────────────────────────────────────────────────────────────┐
//!   │ Key exists?                                                          │
//!   │   YES → update value, set ref=1                                       │
//!   │   NO  → scan from hand until ref=0 slot found                         │
//!   └──────────────────────────────────────────────────────────────────────┘
//!        │
//!        ▼
//!   ┌──────────────────────────────────────────────────────────────────────┐
//!   │ At each slot:                                                        │
//!   │   ref=1 → clear ref, advance hand                                    │
//!   │   ref=0 → evict entry, replace slot, advance hand                    │
//!   └──────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Entry Structure
//!
//! ```text
//!   Entry<K, V>
//!   ┌───────────────────────────────┐
//!   │ key: K                        │
//!   │ value: V                      │
//!   │ referenced: bool              │
//!   └───────────────────────────────┘
//! ```
//!
//! ## Performance Characteristics
//!
//! | Operation  | Time        | Notes                                   |
//! |-----------|-------------|-----------------------------------------|
//! | `insert`  | O(1) amort. | Bounded scan with reference clearing     |
//! | `get`     | O(1)        | Sets reference bit                       |
//! | `touch`   | O(1)        | Sets reference bit                       |
//! | `remove`  | O(1)        | Clears slot + index entry                |
//!
//! ## Notes
//! - Slots are reused in place; keys map directly to slot indices.
//! - `debug_validate_invariants()` is available in debug/test builds.
use std::collections::HashMap;
use std::hash::Hash;

#[derive(Debug)]
struct Entry<K, V> {
    key: K,
    value: V,
    referenced: bool,
}

#[derive(Debug)]
pub struct ClockRing<K, V> {
    slots: Vec<Option<Entry<K, V>>>,
    index: HashMap<K, usize>,
    hand: usize,
    len: usize,
}

impl<K, V> ClockRing<K, V>
where
    K: Eq + Hash + Clone,
{
    pub fn new(capacity: usize) -> Self {
        let mut slots = Vec::with_capacity(capacity);
        slots.resize_with(capacity, || None);
        Self {
            slots,
            index: HashMap::with_capacity(capacity),
            hand: 0,
            len: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn contains(&self, key: &K) -> bool {
        self.index.contains_key(key)
    }

    pub fn peek(&self, key: &K) -> Option<&V> {
        let idx = *self.index.get(key)?;
        self.slots.get(idx)?.as_ref().map(|entry| &entry.value)
    }

    pub fn get(&mut self, key: &K) -> Option<&V> {
        let idx = *self.index.get(key)?;
        let entry = self.slots.get_mut(idx)?.as_mut()?;
        entry.referenced = true;
        Some(&entry.value)
    }

    pub fn touch(&mut self, key: &K) -> bool {
        let idx = match self.index.get(key) {
            Some(idx) => *idx,
            None => return false,
        };
        if let Some(entry) = self.slots.get_mut(idx).and_then(|slot| slot.as_mut()) {
            entry.referenced = true;
            return true;
        }
        false
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<(K, V)> {
        if self.capacity() == 0 {
            return None;
        }

        if let Some(&idx) = self.index.get(&key) {
            if let Some(entry) = self.slots.get_mut(idx).and_then(|slot| slot.as_mut()) {
                entry.value = value;
                entry.referenced = true;
            }
            return None;
        }

        loop {
            let idx = self.hand;
            if let Some(entry) = self.slots.get_mut(idx).and_then(|slot| slot.as_mut()) {
                if entry.referenced {
                    entry.referenced = false;
                    self.advance_hand();
                    continue;
                }

                let evicted = self.slots[idx].take().expect("occupied slot missing");
                self.index.remove(&evicted.key);

                let entry_key = key.clone();
                self.slots[idx] = Some(Entry {
                    key: entry_key,
                    value,
                    referenced: false,
                });
                self.index.insert(key, idx);
                self.advance_hand();
                return Some((evicted.key, evicted.value));
            }

            let entry_key = key.clone();
            self.slots[idx] = Some(Entry {
                key: entry_key,
                value,
                referenced: false,
            });
            self.index.insert(key, idx);
            self.len += 1;
            self.advance_hand();
            return None;
        }
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        let idx = self.index.remove(key)?;
        let entry = self.slots.get_mut(idx)?.take()?;
        self.len -= 1;
        Some(entry.value)
    }

    fn advance_hand(&mut self) {
        let cap = self.capacity();
        if cap == 0 {
            self.hand = 0;
        } else {
            self.hand = (self.hand + 1) % cap;
        }
    }

    #[cfg(any(test, debug_assertions))]
    pub fn debug_validate_invariants(&self) {
        let slot_count = self.slots.iter().filter(|slot| slot.is_some()).count();
        assert_eq!(self.len, slot_count);
        assert_eq!(self.len, self.index.len());

        if self.capacity() == 0 {
            assert_eq!(self.hand, 0);
        } else {
            assert!(self.hand < self.capacity());
        }

        for (key, &idx) in &self.index {
            assert!(idx < self.slots.len());
            let entry = self.slots[idx]
                .as_ref()
                .expect("index points to empty slot");
            assert!(&entry.key == key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_ring_eviction_prefers_unreferenced() {
        let mut ring = ClockRing::new(2);
        ring.insert("a", 1);
        ring.insert("b", 2);
        ring.touch(&"a");
        let evicted = ring.insert("c", 3);

        assert_eq!(evicted, Some(("b", 2)));
        assert!(ring.contains(&"a"));
        assert!(ring.contains(&"c"));
    }

    #[test]
    fn clock_ring_zero_capacity_is_noop() {
        let mut ring = ClockRing::<&str, i32>::new(0);
        assert!(ring.is_empty());
        assert_eq!(ring.capacity(), 0);
        assert_eq!(ring.insert("a", 1), None);
        assert!(ring.is_empty());
        assert!(ring.peek(&"a").is_none());
        assert!(ring.get(&"a").is_none());
        assert!(!ring.contains(&"a"));
    }

    #[test]
    fn clock_ring_insert_and_peek_no_eviction() {
        let mut ring = ClockRing::new(3);
        assert_eq!(ring.insert("a", 1), None);
        assert_eq!(ring.insert("b", 2), None);
        assert_eq!(ring.insert("c", 3), None);
        assert_eq!(ring.len(), 3);
        assert!(ring.contains(&"a"));
        assert!(ring.contains(&"b"));
        assert!(ring.contains(&"c"));
        assert_eq!(ring.peek(&"a"), Some(&1));
        assert_eq!(ring.peek(&"b"), Some(&2));
        assert_eq!(ring.peek(&"c"), Some(&3));
    }

    #[test]
    fn clock_ring_update_existing_key_does_not_grow() {
        let mut ring = ClockRing::new(2);
        assert_eq!(ring.insert("a", 1), None);
        assert_eq!(ring.insert("b", 2), None);
        assert_eq!(ring.len(), 2);

        assert_eq!(ring.insert("a", 10), None);
        assert_eq!(ring.len(), 2);
        assert_eq!(ring.peek(&"a"), Some(&10));
    }

    #[test]
    fn clock_ring_get_sets_referenced_and_eviction_skips_it() {
        let mut ring = ClockRing::new(2);
        ring.insert("a", 1);
        ring.insert("b", 2);
        assert_eq!(ring.get(&"a"), Some(&1));
        let evicted = ring.insert("c", 3);

        assert_eq!(evicted, Some(("b", 2)));
        assert!(ring.contains(&"a"));
        assert!(ring.contains(&"c"));
        assert!(!ring.contains(&"b"));
    }

    #[test]
    fn clock_ring_touch_marks_referenced() {
        let mut ring = ClockRing::new(2);
        ring.insert("a", 1);
        ring.insert("b", 2);
        assert!(ring.touch(&"b"));

        let evicted = ring.insert("c", 3);
        assert_eq!(evicted, Some(("a", 1)));
        assert!(ring.contains(&"b"));
        assert!(ring.contains(&"c"));
    }

    #[test]
    fn clock_ring_remove_clears_slot_and_updates_len() {
        let mut ring = ClockRing::new(3);
        ring.insert("a", 1);
        ring.insert("b", 2);
        ring.insert("c", 3);
        assert_eq!(ring.len(), 3);

        assert_eq!(ring.remove(&"b"), Some(2));
        assert_eq!(ring.len(), 2);
        assert!(!ring.contains(&"b"));
        assert!(ring.peek(&"b").is_none());

        let evicted = ring.insert("d", 4);
        assert!(ring.contains(&"d"));
        assert!(!ring.contains(&"b"));
        if evicted.is_some() {
            assert_eq!(ring.len(), 2);
        } else {
            assert_eq!(ring.len(), 3);
        }
    }

    #[test]
    fn clock_ring_eviction_cycles_with_hand_wrap() {
        let mut ring = ClockRing::new(2);
        ring.insert("a", 1);
        ring.insert("b", 2);

        let evicted1 = ring.insert("c", 3);
        assert!(matches!(evicted1, Some(("a", 1)) | Some(("b", 2))));
        assert_eq!(ring.len(), 2);

        let evicted2 = ring.insert("d", 4);
        assert!(matches!(
            evicted2,
            Some(("a", 1)) | Some(("b", 2)) | Some(("c", 3))
        ));
        assert_eq!(ring.len(), 2);
        assert!(ring.contains(&"d"));
    }

    #[test]
    fn clock_ring_debug_invariants_hold() {
        let mut ring = ClockRing::new(3);
        ring.insert("a", 1);
        ring.insert("b", 2);
        ring.get(&"a");
        ring.insert("c", 3);
        ring.remove(&"b");
        ring.debug_validate_invariants();
    }
}
