use std::collections::HashMap;
use std::hash::Hash;

use crate::ds::intrusive_list::IntrusiveList;
use crate::ds::slot_arena::SlotId;

#[derive(Debug)]
pub struct GhostList<K> {
    list: IntrusiveList<K>,
    index: HashMap<K, SlotId>,
    capacity: usize,
}

impl<K> GhostList<K>
where
    K: Eq + Hash + Clone,
{
    pub fn new(capacity: usize) -> Self {
        Self {
            list: IntrusiveList::with_capacity(capacity),
            index: HashMap::with_capacity(capacity),
            capacity,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.list.len()
    }

    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }

    pub fn contains(&self, key: &K) -> bool {
        self.index.contains_key(key)
    }

    pub fn record(&mut self, key: K) {
        if self.capacity == 0 {
            return;
        }

        if let Some(&id) = self.index.get(&key) {
            self.list.move_to_front(id);
            return;
        }

        if self.list.len() >= self.capacity
            && let Some(old_key) = self.list.pop_back()
        {
            self.index.remove(&old_key);
        }

        let id = self.list.push_front(key.clone());
        self.index.insert(key, id);
    }

    pub fn remove(&mut self, key: &K) -> bool {
        let id = match self.index.remove(key) {
            Some(id) => id,
            None => return false,
        };
        self.list.remove(id);
        true
    }

    pub fn clear(&mut self) {
        self.list.clear();
        self.index.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ghost_list_records_and_evictions() {
        let mut ghost = GhostList::new(2);
        ghost.record("a");
        ghost.record("b");
        assert!(ghost.contains(&"a"));
        assert!(ghost.contains(&"b"));

        ghost.record("a");
        ghost.record("c");

        assert!(ghost.contains(&"a"));
        assert!(ghost.contains(&"c"));
        assert!(!ghost.contains(&"b"));
    }
}
