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
}
