//! Intrusive doubly linked list backed by `SlotArena`.
//!
//! Stores list nodes in a `SlotArena` and links them by `SlotId`, enabling
//! stable handles and O(1) splice/move operations without pointer chasing.
//!
//! ## Architecture
//!
//! ```text
//!   arena (SlotArena<Node<T>>)
//!   ┌────────┬─────────────────────────────────────────────┐
//!   │ SlotId │ Node { value, prev, next }                  │
//!   ├────────┼─────────────────────────────────────────────┤
//!   │ id_1   │ { value: A, prev: None, next: Some(id_2) }  │
//!   │ id_2   │ { value: B, prev: Some(id_1), next: id_3 }  │
//!   │ id_3   │ { value: C, prev: Some(id_2), next: None }  │
//!   └────────┴─────────────────────────────────────────────┘
//!
//!   head ─► [id_1] ◄──► [id_2] ◄──► [id_3] ◄── tail
//! ```
//!
//! ## Operations
//! - `move_to_front(id)`: detach + attach to head
//! - `move_to_back(id)`: detach + attach to tail
//! - `remove(id)`: detach + free slot in arena
//!
//! ## Performance
//! - `push_front` / `push_back`: O(1)
//! - `pop_front` / `pop_back`: O(1)
//! - `move_to_front` / `move_to_back`: O(1)
//! - `iter`: O(n)
//!
//! `debug_validate_invariants()` is available in debug/test builds.
use parking_lot::RwLock;

use crate::ds::slot_arena::{SlotArena, SlotId};

#[derive(Debug)]
struct Node<T> {
    value: T,
    prev: Option<SlotId>,
    next: Option<SlotId>,
    epoch: u64,
}

#[derive(Debug)]
/// Intrusive list that stores nodes in a `SlotArena` and links them via `SlotId`.
pub struct IntrusiveList<T> {
    arena: SlotArena<Node<T>>,
    head: Option<SlotId>,
    tail: Option<SlotId>,
}

impl<T> IntrusiveList<T> {
    /// Creates an empty list.
    pub fn new() -> Self {
        Self {
            arena: SlotArena::new(),
            head: None,
            tail: None,
        }
    }

    /// Creates an empty list with reserved node capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            arena: SlotArena::with_capacity(capacity),
            head: None,
            tail: None,
        }
    }

    /// Returns the number of nodes in the list.
    pub fn len(&self) -> usize {
        self.arena.len()
    }

    /// Returns `true` if the list is empty.
    pub fn is_empty(&self) -> bool {
        self.arena.is_empty()
    }

    /// Returns `true` if `id` is currently a node in this list.
    pub fn contains(&self, id: SlotId) -> bool {
        self.arena.contains(id)
    }

    /// Returns the value at the front (MRU) of the list.
    pub fn front(&self) -> Option<&T> {
        self.head
            .and_then(|id| self.arena.get(id).map(|node| &node.value))
    }

    /// Returns the SlotId at the front (MRU) of the list.
    pub fn front_id(&self) -> Option<SlotId> {
        self.head
    }

    /// Returns the value at the back (LRU) of the list.
    pub fn back(&self) -> Option<&T> {
        self.tail
            .and_then(|id| self.arena.get(id).map(|node| &node.value))
    }

    /// Returns the SlotId at the back (LRU) of the list.
    pub fn back_id(&self) -> Option<SlotId> {
        self.tail
    }

    /// Returns an iterator from front to back.
    pub fn iter(&self) -> IntrusiveListIter<'_, T> {
        IntrusiveListIter {
            list: self,
            current: self.head,
        }
    }

    /// Returns an iterator of SlotIds from front to back.
    pub fn iter_ids(&self) -> IntrusiveListIdIter<'_, T> {
        IntrusiveListIdIter {
            list: self,
            current: self.head,
        }
    }

    /// Returns an iterator of `(SlotId, &T)` from front to back.
    pub fn iter_entries(&self) -> IntrusiveListEntryIter<'_, T> {
        IntrusiveListEntryIter {
            list: self,
            current: self.head,
        }
    }

    /// Returns the value for a node id, if present.
    pub fn get(&self, id: SlotId) -> Option<&T> {
        self.arena.get(id).map(|node| &node.value)
    }

    /// Returns a mutable reference to a node value, if present.
    pub fn get_mut(&mut self, id: SlotId) -> Option<&mut T> {
        self.arena.get_mut(id).map(|node| &mut node.value)
    }

    /// Inserts a new node at the front and returns its `SlotId`.
    pub fn push_front(&mut self, value: T) -> SlotId {
        self.push_front_with_epoch(value, 0)
    }

    /// Inserts a new node at the front with an epoch and returns its `SlotId`.
    pub fn push_front_with_epoch(&mut self, value: T, epoch: u64) -> SlotId {
        let id = self.arena.insert(Node {
            value,
            prev: None,
            next: self.head,
            epoch,
        });
        if let Some(head) = self.head {
            if let Some(node) = self.arena.get_mut(head) {
                node.prev = Some(id);
            }
        } else {
            self.tail = Some(id);
        }
        self.head = Some(id);
        id
    }

    /// Inserts a new node at the back and returns its `SlotId`.
    pub fn push_back(&mut self, value: T) -> SlotId {
        self.push_back_with_epoch(value, 0)
    }

    /// Inserts a new node at the back with an epoch and returns its `SlotId`.
    pub fn push_back_with_epoch(&mut self, value: T, epoch: u64) -> SlotId {
        let id = self.arena.insert(Node {
            value,
            prev: self.tail,
            next: None,
            epoch,
        });
        if let Some(tail) = self.tail {
            if let Some(node) = self.arena.get_mut(tail) {
                node.next = Some(id);
            }
        } else {
            self.head = Some(id);
        }
        self.tail = Some(id);
        id
    }

    /// Returns the epoch recorded for `id`, if present.
    pub fn epoch(&self, id: SlotId) -> Option<u64> {
        self.arena.get(id).map(|node| node.epoch)
    }

    /// Sets the epoch for `id`; returns `false` if `id` is not present.
    pub fn set_epoch(&mut self, id: SlotId, epoch: u64) -> bool {
        if let Some(node) = self.arena.get_mut(id) {
            node.epoch = epoch;
            true
        } else {
            false
        }
    }

    /// Removes and returns the front value.
    pub fn pop_front(&mut self) -> Option<T> {
        let id = self.head?;
        self.detach(id)?;
        self.arena.remove(id).map(|node| node.value)
    }

    /// Removes and returns the back value.
    pub fn pop_back(&mut self) -> Option<T> {
        let id = self.tail?;
        self.detach(id)?;
        self.arena.remove(id).map(|node| node.value)
    }

    /// Removes the node `id` from the list and returns its value.
    pub fn remove(&mut self, id: SlotId) -> Option<T> {
        self.detach(id)?;
        self.arena.remove(id).map(|node| node.value)
    }

    /// Moves an existing node to the front; returns `false` if `id` is not present.
    pub fn move_to_front(&mut self, id: SlotId) -> bool {
        if !self.arena.contains(id) {
            return false;
        }
        if Some(id) == self.head {
            return true;
        }
        self.detach(id);
        self.attach_front(id);
        true
    }

    /// Moves an existing node to the back; returns `false` if `id` is not present.
    pub fn move_to_back(&mut self, id: SlotId) -> bool {
        if !self.arena.contains(id) {
            return false;
        }
        if Some(id) == self.tail {
            return true;
        }
        self.detach(id);
        self.attach_back(id);
        true
    }

    /// Clears the list and frees all nodes.
    pub fn clear(&mut self) {
        self.arena.clear();
        self.head = None;
        self.tail = None;
    }

    /// Clears the list and shrinks internal storage.
    pub fn clear_shrink(&mut self) {
        self.clear();
        self.arena.shrink_to_fit();
    }

    /// Returns an approximate memory footprint in bytes.
    pub fn approx_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.arena.approx_bytes()
    }

    #[cfg(any(test, debug_assertions))]
    /// Returns the list order as SlotIds from head to tail.
    pub fn debug_snapshot_ids(&self) -> Vec<SlotId> {
        self.iter_ids().collect()
    }

    #[cfg(any(test, debug_assertions))]
    /// Returns SlotIds sorted by index for deterministic snapshots.
    pub fn debug_snapshot_ids_sorted(&self) -> Vec<SlotId> {
        let mut ids: Vec<_> = self.iter_ids().collect();
        ids.sort_by_key(|id| id.index());
        ids
    }

    fn detach(&mut self, id: SlotId) -> Option<()> {
        let (prev, next) = {
            let node = self.arena.get(id)?;
            (node.prev, node.next)
        };

        if let Some(prev_id) = prev {
            if let Some(prev_node) = self.arena.get_mut(prev_id) {
                prev_node.next = next;
            }
        } else {
            self.head = next;
        }

        if let Some(next_id) = next {
            if let Some(next_node) = self.arena.get_mut(next_id) {
                next_node.prev = prev;
            }
        } else {
            self.tail = prev;
        }

        if let Some(node) = self.arena.get_mut(id) {
            node.prev = None;
            node.next = None;
        }

        Some(())
    }

    fn attach_front(&mut self, id: SlotId) -> Option<()> {
        let old_head = self.head;
        if let Some(node) = self.arena.get_mut(id) {
            node.prev = None;
            node.next = old_head;
        } else {
            return None;
        }
        if let Some(old_head) = old_head {
            if let Some(head_node) = self.arena.get_mut(old_head) {
                head_node.prev = Some(id);
            }
        } else {
            self.tail = Some(id);
        }
        self.head = Some(id);
        Some(())
    }

    fn attach_back(&mut self, id: SlotId) -> Option<()> {
        let old_tail = self.tail;
        if let Some(node) = self.arena.get_mut(id) {
            node.next = None;
            node.prev = old_tail;
        } else {
            return None;
        }
        if let Some(old_tail) = old_tail {
            if let Some(tail_node) = self.arena.get_mut(old_tail) {
                tail_node.next = Some(id);
            }
        } else {
            self.head = Some(id);
        }
        self.tail = Some(id);
        Some(())
    }

    #[cfg(any(test, debug_assertions))]
    pub fn debug_validate_invariants(&self) {
        if self.head.is_none() || self.tail.is_none() {
            assert!(self.head.is_none());
            assert!(self.tail.is_none());
            assert_eq!(self.len(), 0);
            return;
        }

        let mut seen = std::collections::HashSet::new();
        let mut count = 0usize;
        let mut current = self.head;
        let mut prev = None;

        while let Some(id) = current {
            assert!(seen.insert(id));
            let node = self.arena.get(id).expect("node missing");
            assert_eq!(node.prev, prev);
            if let Some(next_id) = node.next {
                let next_node = self.arena.get(next_id).expect("next node missing");
                assert_eq!(next_node.prev, Some(id));
            } else {
                assert_eq!(self.tail, Some(id));
            }

            prev = Some(id);
            current = node.next;
            count += 1;
            assert!(count <= self.len());
        }

        assert_eq!(count, self.len());
        assert_eq!(self.arena.len(), self.len());
    }
}

pub struct IntrusiveListIter<'a, T> {
    list: &'a IntrusiveList<T>,
    current: Option<SlotId>,
}

impl<'a, T> Iterator for IntrusiveListIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let id = self.current?;
        let node = self.list.arena.get(id)?;
        self.current = node.next;
        Some(&node.value)
    }
}

/// Iterator over SlotIds from front to back.
pub struct IntrusiveListIdIter<'a, T> {
    list: &'a IntrusiveList<T>,
    current: Option<SlotId>,
}

impl<'a, T> Iterator for IntrusiveListIdIter<'a, T> {
    type Item = SlotId;

    fn next(&mut self) -> Option<Self::Item> {
        let id = self.current?;
        let node = self.list.arena.get(id)?;
        self.current = node.next;
        Some(id)
    }
}

/// Iterator over `(SlotId, &T)` pairs from front to back.
pub struct IntrusiveListEntryIter<'a, T> {
    list: &'a IntrusiveList<T>,
    current: Option<SlotId>,
}

impl<'a, T> Iterator for IntrusiveListEntryIter<'a, T> {
    type Item = (SlotId, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        let id = self.current?;
        let node = self.list.arena.get(id)?;
        self.current = node.next;
        Some((id, &node.value))
    }
}

impl<T> Default for IntrusiveList<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
/// Thread-safe wrapper around `IntrusiveList` using a `parking_lot::RwLock`.
pub struct ConcurrentIntrusiveList<T> {
    inner: RwLock<IntrusiveList<T>>,
}

impl<T> ConcurrentIntrusiveList<T> {
    /// Creates an empty concurrent list.
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(IntrusiveList::new()),
        }
    }

    /// Creates an empty concurrent list with reserved node capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: RwLock::new(IntrusiveList::with_capacity(capacity)),
        }
    }

    /// Returns the number of nodes in the list.
    pub fn len(&self) -> usize {
        let list = self.inner.read();
        list.len()
    }

    /// Returns `true` if the list is empty.
    pub fn is_empty(&self) -> bool {
        let list = self.inner.read();
        list.is_empty()
    }

    /// Returns `true` if `id` is currently a node in this list.
    pub fn contains(&self, id: SlotId) -> bool {
        let list = self.inner.read();
        list.contains(id)
    }

    /// Inserts a value at the front and returns its `SlotId`.
    pub fn push_front(&self, value: T) -> SlotId {
        let mut list = self.inner.write();
        list.push_front(value)
    }

    /// Tries to insert a value at the front without blocking.
    pub fn try_push_front(&self, value: T) -> Option<SlotId> {
        let mut list = self.inner.try_write()?;
        Some(list.push_front(value))
    }

    /// Inserts a value at the front with an epoch and returns its `SlotId`.
    pub fn push_front_with_epoch(&self, value: T, epoch: u64) -> SlotId {
        let mut list = self.inner.write();
        list.push_front_with_epoch(value, epoch)
    }

    /// Tries to insert a value at the front with an epoch without blocking.
    pub fn try_push_front_with_epoch(&self, value: T, epoch: u64) -> Option<SlotId> {
        let mut list = self.inner.try_write()?;
        Some(list.push_front_with_epoch(value, epoch))
    }

    /// Inserts a value at the back and returns its `SlotId`.
    pub fn push_back(&self, value: T) -> SlotId {
        let mut list = self.inner.write();
        list.push_back(value)
    }

    /// Tries to insert a value at the back without blocking.
    pub fn try_push_back(&self, value: T) -> Option<SlotId> {
        let mut list = self.inner.try_write()?;
        Some(list.push_back(value))
    }

    /// Inserts a value at the back with an epoch and returns its `SlotId`.
    pub fn push_back_with_epoch(&self, value: T, epoch: u64) -> SlotId {
        let mut list = self.inner.write();
        list.push_back_with_epoch(value, epoch)
    }

    /// Tries to insert a value at the back with an epoch without blocking.
    pub fn try_push_back_with_epoch(&self, value: T, epoch: u64) -> Option<SlotId> {
        let mut list = self.inner.try_write()?;
        Some(list.push_back_with_epoch(value, epoch))
    }

    /// Removes and returns the front value.
    pub fn pop_front(&self) -> Option<T> {
        let mut list = self.inner.write();
        list.pop_front()
    }

    /// Tries to remove and return the front value without blocking.
    pub fn try_pop_front(&self) -> Option<Option<T>> {
        let mut list = self.inner.try_write()?;
        Some(list.pop_front())
    }

    /// Removes and returns the back value.
    pub fn pop_back(&self) -> Option<T> {
        let mut list = self.inner.write();
        list.pop_back()
    }

    /// Tries to remove and return the back value without blocking.
    pub fn try_pop_back(&self) -> Option<Option<T>> {
        let mut list = self.inner.try_write()?;
        Some(list.pop_back())
    }

    /// Removes the node `id` and returns its value, if present.
    pub fn remove(&self, id: SlotId) -> Option<T> {
        let mut list = self.inner.write();
        list.remove(id)
    }

    /// Tries to remove the node `id` without blocking.
    pub fn try_remove(&self, id: SlotId) -> Option<Option<T>> {
        let mut list = self.inner.try_write()?;
        Some(list.remove(id))
    }

    /// Moves an existing node to the front; returns `false` if not present.
    pub fn move_to_front(&self, id: SlotId) -> bool {
        let mut list = self.inner.write();
        list.move_to_front(id)
    }

    /// Tries to move a node to the front without blocking.
    pub fn try_move_to_front(&self, id: SlotId) -> Option<bool> {
        let mut list = self.inner.try_write()?;
        Some(list.move_to_front(id))
    }

    /// Moves an existing node to the back; returns `false` if not present.
    pub fn move_to_back(&self, id: SlotId) -> bool {
        let mut list = self.inner.write();
        list.move_to_back(id)
    }

    /// Tries to move a node to the back without blocking.
    pub fn try_move_to_back(&self, id: SlotId) -> Option<bool> {
        let mut list = self.inner.try_write()?;
        Some(list.move_to_back(id))
    }

    /// Runs `f` on a shared reference to the value at `id`, if present.
    pub fn get_with<R>(&self, id: SlotId, f: impl FnOnce(&T) -> R) -> Option<R> {
        let list = self.inner.read();
        list.get(id).map(f)
    }

    /// Tries to run `f` on a shared reference to the value at `id` without blocking.
    pub fn try_get_with<R>(&self, id: SlotId, f: impl FnOnce(&T) -> R) -> Option<R> {
        let list = self.inner.try_read()?;
        list.get(id).map(f)
    }

    /// Runs `f` on a mutable reference to the value at `id`, if present.
    pub fn get_mut_with<R>(&self, id: SlotId, f: impl FnOnce(&mut T) -> R) -> Option<R> {
        let mut list = self.inner.write();
        list.get_mut(id).map(f)
    }

    /// Tries to run `f` on a mutable reference to the value at `id` without blocking.
    pub fn try_get_mut_with<R>(&self, id: SlotId, f: impl FnOnce(&mut T) -> R) -> Option<R> {
        let mut list = self.inner.try_write()?;
        list.get_mut(id).map(f)
    }

    /// Runs `f` on a shared reference to the front value, if present.
    pub fn front_with<R>(&self, f: impl FnOnce(&T) -> R) -> Option<R> {
        let list = self.inner.read();
        list.front().map(f)
    }

    /// Tries to run `f` on a shared reference to the front value without blocking.
    pub fn try_front_with<R>(&self, f: impl FnOnce(&T) -> R) -> Option<R> {
        let list = self.inner.try_read()?;
        list.front().map(f)
    }

    /// Runs `f` on a shared reference to the back value, if present.
    pub fn back_with<R>(&self, f: impl FnOnce(&T) -> R) -> Option<R> {
        let list = self.inner.read();
        list.back().map(f)
    }

    /// Tries to run `f` on a shared reference to the back value without blocking.
    pub fn try_back_with<R>(&self, f: impl FnOnce(&T) -> R) -> Option<R> {
        let list = self.inner.try_read()?;
        list.back().map(f)
    }

    /// Returns the epoch recorded for `id`, if present.
    pub fn epoch(&self, id: SlotId) -> Option<u64> {
        let list = self.inner.read();
        list.epoch(id)
    }

    /// Tries to read the epoch for `id` without blocking.
    pub fn try_epoch(&self, id: SlotId) -> Option<Option<u64>> {
        let list = self.inner.try_read()?;
        Some(list.epoch(id))
    }

    /// Sets the epoch for `id`; returns `false` if `id` is not present.
    pub fn set_epoch(&self, id: SlotId, epoch: u64) -> bool {
        let mut list = self.inner.write();
        list.set_epoch(id, epoch)
    }

    /// Tries to set the epoch for `id` without blocking.
    pub fn try_set_epoch(&self, id: SlotId, epoch: u64) -> Option<bool> {
        let mut list = self.inner.try_write()?;
        Some(list.set_epoch(id, epoch))
    }

    /// Clears the list.
    pub fn clear(&self) {
        let mut list = self.inner.write();
        list.clear();
    }

    /// Tries to clear the list without blocking.
    pub fn try_clear(&self) -> bool {
        if let Some(mut list) = self.inner.try_write() {
            list.clear();
            true
        } else {
            false
        }
    }

    /// Clears the list and shrinks internal storage.
    pub fn clear_shrink(&self) {
        let mut list = self.inner.write();
        list.clear_shrink();
    }

    /// Tries to clear and shrink without blocking.
    pub fn try_clear_shrink(&self) -> bool {
        if let Some(mut list) = self.inner.try_write() {
            list.clear_shrink();
            true
        } else {
            false
        }
    }

    /// Returns an approximate memory footprint in bytes.
    pub fn approx_bytes(&self) -> usize {
        let list = self.inner.read();
        list.approx_bytes()
    }
}

impl<T> Default for ConcurrentIntrusiveList<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intrusive_list_basic_ops() {
        let mut list = IntrusiveList::new();
        let a = list.push_front("a");
        let b = list.push_back("b");
        let c = list.push_back("c");

        assert_eq!(list.front(), Some(&"a"));
        assert_eq!(list.back(), Some(&"c"));
        assert_eq!(list.len(), 3);

        assert!(list.move_to_front(c));
        assert_eq!(list.front(), Some(&"c"));
        assert_eq!(list.back(), Some(&"b"));

        assert_eq!(list.remove(b), Some("b"));
        assert_eq!(list.len(), 2);

        assert_eq!(list.pop_front(), Some("c"));
        assert_eq!(list.pop_back(), Some("a"));
        assert!(list.is_empty());

        assert!(!list.contains(a));
    }

    #[test]
    fn concurrent_intrusive_list_basic_ops() {
        let list = ConcurrentIntrusiveList::new();
        let a = list.push_front("a");
        let b = list.push_back("b");
        assert_eq!(list.front_with(|v| *v), Some("a"));
        assert_eq!(list.back_with(|v| *v), Some("b"));
        assert_eq!(list.len(), 2);

        assert!(list.move_to_front(b));
        assert_eq!(list.front_with(|v| *v), Some("b"));
        assert_eq!(list.pop_back(), Some("a"));
        assert_eq!(list.pop_back(), Some("b"));

        assert!(list.is_empty());
        assert!(!list.contains(a));
    }

    #[test]
    fn intrusive_list_iter_order() {
        let mut list = IntrusiveList::new();
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        let values: Vec<_> = list.iter().copied().collect();
        assert_eq!(values, vec![1, 2, 3]);
    }

    #[test]
    fn intrusive_list_move_to_front_back_edges() {
        let mut list = IntrusiveList::new();
        let a = list.push_back("a");
        let b = list.push_back("b");
        let c = list.push_back("c");

        assert!(list.move_to_front(a));
        let values: Vec<_> = list.iter().copied().collect();
        assert_eq!(values, vec!["a", "b", "c"]);

        assert!(list.move_to_back(a));
        let values: Vec<_> = list.iter().copied().collect();
        assert_eq!(values, vec!["b", "c", "a"]);

        assert!(list.move_to_front(c));
        let values: Vec<_> = list.iter().copied().collect();
        assert_eq!(values, vec!["c", "b", "a"]);

        assert!(list.contains(b));
    }

    #[test]
    fn intrusive_list_remove_middle_and_ends() {
        let mut list = IntrusiveList::new();
        let a = list.push_back("a");
        let b = list.push_back("b");
        let c = list.push_back("c");

        assert_eq!(list.remove(b), Some("b"));
        let values: Vec<_> = list.iter().copied().collect();
        assert_eq!(values, vec!["a", "c"]);

        assert_eq!(list.remove(a), Some("a"));
        assert_eq!(list.front(), Some(&"c"));
        assert_eq!(list.back(), Some(&"c"));

        assert_eq!(list.remove(c), Some("c"));
        assert!(list.is_empty());
        assert_eq!(list.front(), None);
        assert_eq!(list.back(), None);
    }

    #[test]
    fn intrusive_list_clear_resets_state() {
        let mut list = IntrusiveList::new();
        list.push_back(1);
        list.push_back(2);
        list.clear();
        assert!(list.is_empty());
        assert_eq!(list.front(), None);
        assert_eq!(list.back(), None);
        assert_eq!(list.pop_front(), None);
        assert_eq!(list.pop_back(), None);
    }

    #[test]
    fn intrusive_list_get_mut_updates_value() {
        let mut list = IntrusiveList::new();
        let id = list.push_back(10);
        if let Some(value) = list.get_mut(id) {
            *value = 20;
        }
        assert_eq!(list.get(id), Some(&20));
    }

    #[test]
    fn intrusive_list_id_and_entry_iters() {
        let mut list = IntrusiveList::new();
        let a = list.push_back("a");
        let b = list.push_back("b");
        let c = list.push_back("c");

        assert_eq!(list.front_id(), Some(a));
        assert_eq!(list.back_id(), Some(c));

        let ids: Vec<_> = list.iter_ids().collect();
        assert_eq!(ids, vec![a, b, c]);

        let entries: Vec<_> = list.iter_entries().map(|(id, v)| (id, *v)).collect();
        assert_eq!(entries, vec![(a, "a"), (b, "b"), (c, "c")]);
    }

    #[test]
    fn intrusive_list_epoch_tracking() {
        let mut list = IntrusiveList::new();
        let a = list.push_front_with_epoch("a", 7);
        let b = list.push_back("b");

        assert_eq!(list.epoch(a), Some(7));
        assert_eq!(list.epoch(b), Some(0));

        assert!(list.set_epoch(b, 9));
        assert_eq!(list.epoch(b), Some(9));

        list.remove(b);
        assert_eq!(list.epoch(b), None);
        assert!(!list.set_epoch(b, 3));
    }

    #[test]
    fn intrusive_list_debug_snapshot_ids() {
        let mut list = IntrusiveList::new();
        let a = list.push_back(1);
        let b = list.push_back(2);
        let c = list.push_back(3);
        assert_eq!(list.debug_snapshot_ids(), vec![a, b, c]);
        assert_eq!(list.debug_snapshot_ids_sorted(), vec![a, b, c]);
    }

    #[test]
    fn concurrent_intrusive_list_clear_and_accessors() {
        let list = ConcurrentIntrusiveList::new();
        let a = list.push_front(1);
        let b = list.push_back(2);

        assert_eq!(list.get_with(a, |v| *v), Some(1));
        assert_eq!(list.get_with(b, |v| *v), Some(2));
        assert!(list.contains(a));
        assert!(list.contains(b));

        list.clear();
        assert!(list.is_empty());
        assert_eq!(list.front_with(|v| *v), None);
        assert_eq!(list.back_with(|v| *v), None);
        assert!(!list.contains(a));
        assert!(!list.contains(b));
    }

    #[test]
    fn concurrent_intrusive_list_try_ops() {
        let list = ConcurrentIntrusiveList::new();
        let a = list.try_push_front(1).unwrap();
        let b = list.try_push_back(2).unwrap();
        assert_eq!(list.try_get_with(a, |v| *v), Some(1));
        assert_eq!(list.try_get_with(b, |v| *v), Some(2));
        assert_eq!(list.try_front_with(|v| *v), Some(1));
        assert!(list.try_move_to_back(a).unwrap());
        assert_eq!(list.try_pop_front().unwrap(), Some(2));
        assert!(list.try_clear());
        assert!(list.is_empty());
    }

    #[test]
    fn concurrent_intrusive_list_epoch_ops() {
        let list = ConcurrentIntrusiveList::new();
        let a = list.push_front_with_epoch(1, 11);
        let b = list.push_back(2);

        assert_eq!(list.epoch(a), Some(11));
        assert_eq!(list.epoch(b), Some(0));
        assert!(list.set_epoch(b, 15));
        assert_eq!(list.epoch(b), Some(15));

        assert_eq!(list.try_epoch(a).unwrap(), Some(11));
        assert!(list.try_set_epoch(a, 20).unwrap());
        assert_eq!(list.epoch(a), Some(20));
    }

    #[test]
    fn intrusive_list_debug_invariants_hold() {
        let mut list = IntrusiveList::new();
        let a = list.push_back(1);
        let b = list.push_back(2);
        let c = list.push_back(3);
        list.move_to_front(b);
        list.remove(a);
        list.remove(c);
        list.debug_validate_invariants();
    }
}
