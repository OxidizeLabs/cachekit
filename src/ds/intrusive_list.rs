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
use crate::ds::slot_arena::{SlotArena, SlotId};
use parking_lot::RwLock;

#[derive(Debug)]
struct Node<T> {
    value: T,
    prev: Option<SlotId>,
    next: Option<SlotId>,
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

    /// Returns the value at the back (LRU) of the list.
    pub fn back(&self) -> Option<&T> {
        self.tail
            .and_then(|id| self.arena.get(id).map(|node| &node.value))
    }

    /// Returns an iterator from front to back.
    pub fn iter(&self) -> IntrusiveListIter<'_, T> {
        IntrusiveListIter {
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
        let id = self.arena.insert(Node {
            value,
            prev: None,
            next: self.head,
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
        let id = self.arena.insert(Node {
            value,
            prev: self.tail,
            next: None,
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

    /// Inserts a value at the back and returns its `SlotId`.
    pub fn push_back(&self, value: T) -> SlotId {
        let mut list = self.inner.write();
        list.push_back(value)
    }

    /// Removes and returns the front value.
    pub fn pop_front(&self) -> Option<T> {
        let mut list = self.inner.write();
        list.pop_front()
    }

    /// Removes and returns the back value.
    pub fn pop_back(&self) -> Option<T> {
        let mut list = self.inner.write();
        list.pop_back()
    }

    /// Removes the node `id` and returns its value, if present.
    pub fn remove(&self, id: SlotId) -> Option<T> {
        let mut list = self.inner.write();
        list.remove(id)
    }

    /// Moves an existing node to the front; returns `false` if not present.
    pub fn move_to_front(&self, id: SlotId) -> bool {
        let mut list = self.inner.write();
        list.move_to_front(id)
    }

    /// Moves an existing node to the back; returns `false` if not present.
    pub fn move_to_back(&self, id: SlotId) -> bool {
        let mut list = self.inner.write();
        list.move_to_back(id)
    }

    /// Runs `f` on a shared reference to the value at `id`, if present.
    pub fn get_with<R>(&self, id: SlotId, f: impl FnOnce(&T) -> R) -> Option<R> {
        let list = self.inner.read();
        list.get(id).map(f)
    }

    /// Runs `f` on a mutable reference to the value at `id`, if present.
    pub fn get_mut_with<R>(&self, id: SlotId, f: impl FnOnce(&mut T) -> R) -> Option<R> {
        let mut list = self.inner.write();
        list.get_mut(id).map(f)
    }

    /// Runs `f` on a shared reference to the front value, if present.
    pub fn front_with<R>(&self, f: impl FnOnce(&T) -> R) -> Option<R> {
        let list = self.inner.read();
        list.front().map(f)
    }

    /// Runs `f` on a shared reference to the back value, if present.
    pub fn back_with<R>(&self, f: impl FnOnce(&T) -> R) -> Option<R> {
        let list = self.inner.read();
        list.back().map(f)
    }

    /// Clears the list.
    pub fn clear(&self) {
        let mut list = self.inner.write();
        list.clear();
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
