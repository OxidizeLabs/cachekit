//! Slot arena with stable `SlotId` handles.
//!
//! Stores elements in a `Vec<Option<T>>` and reuses freed slots via a free list.
//! This keeps indices stable and avoids per-operation allocation.
//!
//! ## Architecture
//!
//! ```text
//!   slots: Vec<Option<T>>
//!   free_list: Vec<usize>
//!
//!   index: 0     1     2     3
//!          [T]  [ ]   [T]   [ ]
//!                 ^         ^
//!                 |         |
//!             free_list = [1, 3]
//! ```
//!
//! ## Operations
//! - `insert(value)`: uses a free slot if available, otherwise grows `slots`
//! - `remove(id)`: clears slot and pushes index to `free_list`
//! - `get(id)`: returns `None` if slot is empty or out of bounds
//!
//! ## Performance
//! - `insert` / `remove` / `get`: O(1) average
//! - `iter`: O(n) over slots
//!
//! `debug_validate_invariants()` is available in debug/test builds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// Stable handle into a `SlotArena`.
///
/// `SlotId` values remain valid until the referenced slot is removed; after
/// removal, the numeric index may be reused by a later `insert`.
pub struct SlotId(pub(crate) usize);

impl SlotId {
    /// Returns the underlying slot index.
    pub fn index(self) -> usize {
        self.0
    }
}

#[derive(Debug)]
/// Arena that stores values in reusable slots and returns stable `SlotId`s.
pub struct SlotArena<T> {
    slots: Vec<Option<T>>,
    free_list: Vec<usize>,
    len: usize,
}

impl<T> SlotArena<T> {
    /// Creates an empty arena.
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_list: Vec::new(),
            len: 0,
        }
    }

    /// Creates an empty arena with reserved capacity for slots.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            slots: Vec::with_capacity(capacity),
            free_list: Vec::new(),
            len: 0,
        }
    }

    /// Inserts a value and returns its `SlotId`.
    pub fn insert(&mut self, value: T) -> SlotId {
        let idx = if let Some(idx) = self.free_list.pop() {
            self.slots[idx] = Some(value);
            idx
        } else {
            self.slots.push(Some(value));
            self.slots.len() - 1
        };
        self.len += 1;
        SlotId(idx)
    }

    /// Removes the value at `id` and returns it, or `None` if empty/out of bounds.
    pub fn remove(&mut self, id: SlotId) -> Option<T> {
        let slot = self.slots.get_mut(id.0)?;
        let value = slot.take()?;
        self.free_list.push(id.0);
        self.len -= 1;
        Some(value)
    }

    /// Returns a shared reference to the value at `id`, if present.
    pub fn get(&self, id: SlotId) -> Option<&T> {
        self.slots.get(id.0).and_then(|slot| slot.as_ref())
    }

    /// Returns a mutable reference to the value at `id`, if present.
    pub fn get_mut(&mut self, id: SlotId) -> Option<&mut T> {
        self.slots.get_mut(id.0).and_then(|slot| slot.as_mut())
    }

    /// Returns `true` if `id` currently refers to a live slot.
    pub fn contains(&self, id: SlotId) -> bool {
        self.slots
            .get(id.0)
            .map(|slot| slot.is_some())
            .unwrap_or(false)
    }

    /// Returns the number of live entries.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the arena has no live entries.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the backing vector capacity (number of slots that can be held without realloc).
    pub fn capacity(&self) -> usize {
        self.slots.capacity()
    }

    /// Removes all entries and resets internal state.
    pub fn clear(&mut self) {
        self.slots.clear();
        self.free_list.clear();
        self.len = 0;
    }

    /// Iterates over live `(SlotId, &T)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (SlotId, &T)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(idx, slot)| slot.as_ref().map(|value| (SlotId(idx), value)))
    }

    #[cfg(any(test, debug_assertions))]
    pub fn debug_validate_invariants(&self) {
        let live_count = self.slots.iter().filter(|slot| slot.is_some()).count();
        assert_eq!(self.len, live_count);
        assert!(self.len <= self.slots.len());

        let mut seen_free = std::collections::HashSet::new();
        for &idx in &self.free_list {
            assert!(idx < self.slots.len());
            assert!(self.slots[idx].is_none());
            assert!(seen_free.insert(idx));
        }

        assert_eq!(self.slots.len(), self.free_list.len() + self.len);
    }
}

impl<T> Default for SlotArena<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
/// Thread-safe wrapper around `SlotArena` using a `parking_lot::RwLock`.
pub struct ConcurrentSlotArena<T> {
    inner: parking_lot::RwLock<SlotArena<T>>,
}

impl<T> ConcurrentSlotArena<T> {
    /// Creates an empty concurrent arena.
    pub fn new() -> Self {
        Self {
            inner: parking_lot::RwLock::new(SlotArena::new()),
        }
    }

    /// Creates an empty concurrent arena with reserved slot capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: parking_lot::RwLock::new(SlotArena::with_capacity(capacity)),
        }
    }

    /// Inserts a value and returns its `SlotId`.
    pub fn insert(&self, value: T) -> SlotId {
        let mut arena = self.inner.write();
        arena.insert(value)
    }

    /// Removes the value at `id` and returns it, if present.
    pub fn remove(&self, id: SlotId) -> Option<T> {
        let mut arena = self.inner.write();
        arena.remove(id)
    }

    /// Runs `f` on a shared reference to the value at `id`, if present.
    pub fn get_with<R>(&self, id: SlotId, f: impl FnOnce(&T) -> R) -> Option<R> {
        let arena = self.inner.read();
        arena.get(id).map(f)
    }

    /// Runs `f` on a mutable reference to the value at `id`, if present.
    pub fn get_mut_with<R>(&self, id: SlotId, f: impl FnOnce(&mut T) -> R) -> Option<R> {
        let mut arena = self.inner.write();
        arena.get_mut(id).map(f)
    }

    /// Returns `true` if `id` currently refers to a live slot.
    pub fn contains(&self, id: SlotId) -> bool {
        let arena = self.inner.read();
        arena.contains(id)
    }

    /// Returns the number of live entries.
    pub fn len(&self) -> usize {
        let arena = self.inner.read();
        arena.len()
    }

    /// Returns `true` if there are no live entries.
    pub fn is_empty(&self) -> bool {
        let arena = self.inner.read();
        arena.is_empty()
    }

    /// Returns the backing vector capacity.
    pub fn capacity(&self) -> usize {
        let arena = self.inner.read();
        arena.capacity()
    }

    /// Clears all entries.
    pub fn clear(&self) {
        let mut arena = self.inner.write();
        arena.clear();
    }
}

impl<T> Default for ConcurrentSlotArena<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_arena_insert_remove_reuse() {
        let mut arena = SlotArena::new();
        let id1 = arena.insert("a");
        let id2 = arena.insert("b");
        assert_eq!(arena.len(), 2);
        assert_eq!(arena.get(id1), Some(&"a"));
        assert_eq!(arena.get(id2), Some(&"b"));

        assert_eq!(arena.remove(id1), Some("a"));
        assert_eq!(arena.len(), 1);

        let id3 = arena.insert("c");
        assert_eq!(arena.len(), 2);
        assert_eq!(arena.get(id3), Some(&"c"));
        assert_eq!(id1.index(), id3.index());
    }

    #[test]
    fn concurrent_slot_arena_basic_ops() {
        let arena = ConcurrentSlotArena::new();
        let id = arena.insert(10);
        assert_eq!(arena.get_with(id, |v| *v), Some(10));
        assert!(arena.contains(id));
        assert_eq!(arena.len(), 1);

        arena.get_mut_with(id, |v| *v = 20);
        assert_eq!(arena.get_with(id, |v| *v), Some(20));
        assert_eq!(arena.remove(id), Some(20));
        assert!(!arena.contains(id));
        assert!(arena.is_empty());
    }

    #[test]
    fn slot_arena_remove_invalid_id_is_none() {
        let mut arena: SlotArena<i32> = SlotArena::new();
        let id = SlotId(0);
        assert_eq!(arena.remove(id), None);
        assert!(!arena.contains(id));
        assert!(arena.is_empty());
    }

    #[test]
    fn slot_arena_clear_resets_state() {
        let mut arena = SlotArena::with_capacity(4);
        let a = arena.insert("a");
        let b = arena.insert("b");
        assert_eq!(arena.len(), 2);
        assert!(arena.contains(a));
        assert!(arena.contains(b));

        arena.clear();
        assert_eq!(arena.len(), 0);
        assert!(arena.is_empty());
        assert!(!arena.contains(a));
        assert!(!arena.contains(b));
        assert_eq!(arena.iter().count(), 0);
    }

    #[test]
    fn slot_arena_iter_skips_empty_slots() {
        let mut arena = SlotArena::new();
        let a = arena.insert("a");
        let b = arena.insert("b");
        let c = arena.insert("c");
        assert_eq!(arena.remove(b), Some("b"));

        let mut values: Vec<_> = arena.iter().map(|(_, v)| *v).collect();
        values.sort();
        assert_eq!(values, vec!["a", "c"]);
        assert!(arena.contains(a));
        assert!(arena.contains(c));
    }

    #[test]
    fn slot_arena_get_mut_updates_value() {
        let mut arena = SlotArena::new();
        let id = arena.insert(1);
        if let Some(value) = arena.get_mut(id) {
            *value = 2;
        }
        assert_eq!(arena.get(id), Some(&2));
    }

    #[test]
    fn slot_arena_capacity_tracking() {
        let arena: SlotArena<i32> = SlotArena::with_capacity(16);
        assert!(arena.capacity() >= 16);
        assert_eq!(arena.len(), 0);
    }

    #[test]
    fn concurrent_slot_arena_clear_and_accessors() {
        let arena = ConcurrentSlotArena::new();
        let a = arena.insert(1);
        let b = arena.insert(2);
        assert_eq!(arena.get_with(a, |v| *v), Some(1));
        assert_eq!(arena.get_with(b, |v| *v), Some(2));

        arena.clear();
        assert!(arena.is_empty());
        assert!(!arena.contains(a));
        assert!(!arena.contains(b));
        assert_eq!(arena.get_with(a, |v| *v), None);
    }

    #[test]
    fn concurrent_slot_arena_get_mut_with_updates_value() {
        let arena = ConcurrentSlotArena::new();
        let id = arena.insert(5);
        arena.get_mut_with(id, |v| *v = 10);
        assert_eq!(arena.get_with(id, |v| *v), Some(10));
    }

    #[test]
    fn slot_arena_debug_invariants_hold() {
        let mut arena = SlotArena::new();
        let a = arena.insert(1);
        let b = arena.insert(2);
        arena.remove(a);
        let _ = arena.insert(3);
        assert!(arena.contains(b));
        arena.debug_validate_invariants();
    }
}
