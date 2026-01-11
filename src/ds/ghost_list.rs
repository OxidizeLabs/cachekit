//! Bounded recency list for ghost entries.
//!
//! Used by adaptive policies (ARC/2Q-style) to track recently evicted keys
//! without storing values. Implemented as an `IntrusiveList` plus an index.
//!
//! ## Architecture
//!
//! ```text
//!   index: HashMap<K, SlotId>          list: IntrusiveList<K>
//!   ┌─────────┬─────────┐              head ─► [A] ◄──► [B] ◄──► [C] ◄── tail
//!   │  key A  │  id_1   │                 MRU                       LRU
//!   │  key B  │  id_2   │
//!   └─────────┴─────────┘
//! ```
//!
//! ## Behavior
//! - `record(k)`: moves key to MRU, evicts LRU if at capacity
//! - `remove(k)`: deletes from list and index
//! - `clear()`: resets both list and index
//!
//! ## Performance
//! - `record` / `remove` / `contains`: O(1) average
//!
//! `debug_validate_invariants()` is available in debug/test builds.
use std::collections::HashMap;
use std::hash::Hash;

use crate::ds::intrusive_list::IntrusiveList;
use crate::ds::slot_arena::SlotId;

#[derive(Debug)]
/// Bounded recency list of keys (no values), typically for ARC-style ghost tracking.
pub struct GhostList<K> {
    list: IntrusiveList<K>,
    index: HashMap<K, SlotId>,
    capacity: usize,
}

impl<K> GhostList<K>
where
    K: Eq + Hash + Clone,
{
    /// Creates a new ghost list with a maximum of `capacity` keys.
    pub fn new(capacity: usize) -> Self {
        Self {
            list: IntrusiveList::with_capacity(capacity),
            index: HashMap::with_capacity(capacity),
            capacity,
        }
    }

    /// Returns the configured capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of keys currently tracked.
    pub fn len(&self) -> usize {
        self.list.len()
    }

    /// Returns `true` if there are no keys tracked.
    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }

    /// Returns `true` if `key` is present.
    pub fn contains(&self, key: &K) -> bool {
        self.index.contains_key(key)
    }

    /// Records `key` as most-recently-seen, evicting the least recent if needed.
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

    /// Removes `key` from the ghost list; returns `true` if it was present.
    pub fn remove(&mut self, key: &K) -> bool {
        let id = match self.index.remove(key) {
            Some(id) => id,
            None => return false,
        };
        self.list.remove(id);
        true
    }

    /// Clears all tracked keys.
    pub fn clear(&mut self) {
        self.list.clear();
        self.index.clear();
    }

    #[cfg(any(test, debug_assertions))]
    pub fn debug_validate_invariants(&self) {
        assert_eq!(self.list.len(), self.index.len());
        assert!(self.list.len() <= self.capacity);
        if self.capacity == 0 {
            assert!(self.list.is_empty());
        }
        for &id in self.index.values() {
            assert!(self.list.contains(id));
        }
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

    #[test]
    fn ghost_list_zero_capacity_is_noop() {
        let mut ghost = GhostList::new(0);
        ghost.record("a");
        ghost.record("b");
        assert!(ghost.is_empty());
        assert_eq!(ghost.len(), 0);
        assert!(!ghost.contains(&"a"));
        assert!(!ghost.contains(&"b"));
    }

    #[test]
    fn ghost_list_record_existing_moves_to_front() {
        let mut ghost = GhostList::new(3);
        ghost.record("a");
        ghost.record("b");
        ghost.record("c");
        assert!(ghost.contains(&"a"));
        assert!(ghost.contains(&"b"));
        assert!(ghost.contains(&"c"));

        ghost.record("a");
        ghost.record("d");

        assert!(ghost.contains(&"a"));
        assert!(!ghost.contains(&"b"));
        assert!(ghost.contains(&"c"));
        assert!(ghost.contains(&"d"));
    }

    #[test]
    fn ghost_list_remove_existing_and_missing() {
        let mut ghost = GhostList::new(2);
        ghost.record("a");
        ghost.record("b");
        assert!(ghost.remove(&"a"));
        assert!(!ghost.contains(&"a"));
        assert_eq!(ghost.len(), 1);

        assert!(!ghost.remove(&"missing"));
        assert_eq!(ghost.len(), 1);
    }

    #[test]
    fn ghost_list_clear_resets_state() {
        let mut ghost = GhostList::new(2);
        ghost.record("a");
        ghost.record("b");
        ghost.clear();

        assert!(ghost.is_empty());
        assert_eq!(ghost.len(), 0);
        assert!(!ghost.contains(&"a"));
        assert!(!ghost.contains(&"b"));
    }

    #[test]
    fn ghost_list_debug_invariants_hold() {
        let mut ghost = GhostList::new(2);
        ghost.record("a");
        ghost.record("b");
        ghost.record("a");
        ghost.debug_validate_invariants();
    }
}
