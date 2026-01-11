#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SlotId(pub(crate) usize);

impl SlotId {
    pub fn index(self) -> usize {
        self.0
    }
}

#[derive(Debug)]
pub struct SlotArena<T> {
    slots: Vec<Option<T>>,
    free_list: Vec<usize>,
    len: usize,
}

impl<T> SlotArena<T> {
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_list: Vec::new(),
            len: 0,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            slots: Vec::with_capacity(capacity),
            free_list: Vec::new(),
            len: 0,
        }
    }

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

    pub fn remove(&mut self, id: SlotId) -> Option<T> {
        let slot = self.slots.get_mut(id.0)?;
        let value = slot.take()?;
        self.free_list.push(id.0);
        self.len -= 1;
        Some(value)
    }

    pub fn get(&self, id: SlotId) -> Option<&T> {
        self.slots.get(id.0).and_then(|slot| slot.as_ref())
    }

    pub fn get_mut(&mut self, id: SlotId) -> Option<&mut T> {
        self.slots.get_mut(id.0).and_then(|slot| slot.as_mut())
    }

    pub fn contains(&self, id: SlotId) -> bool {
        self.slots
            .get(id.0)
            .map(|slot| slot.is_some())
            .unwrap_or(false)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.slots.capacity()
    }

    pub fn clear(&mut self) {
        self.slots.clear();
        self.free_list.clear();
        self.len = 0;
    }

    pub fn iter(&self) -> impl Iterator<Item = (SlotId, &T)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(idx, slot)| slot.as_ref().map(|value| (SlotId(idx), value)))
    }
}

impl<T> Default for SlotArena<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct ConcurrentSlotArena<T> {
    inner: parking_lot::RwLock<SlotArena<T>>,
}

impl<T> ConcurrentSlotArena<T> {
    pub fn new() -> Self {
        Self {
            inner: parking_lot::RwLock::new(SlotArena::new()),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: parking_lot::RwLock::new(SlotArena::with_capacity(capacity)),
        }
    }

    pub fn insert(&self, value: T) -> SlotId {
        let mut arena = self.inner.write();
        arena.insert(value)
    }

    pub fn remove(&self, id: SlotId) -> Option<T> {
        let mut arena = self.inner.write();
        arena.remove(id)
    }

    pub fn get_with<R>(&self, id: SlotId, f: impl FnOnce(&T) -> R) -> Option<R> {
        let arena = self.inner.read();
        arena.get(id).map(f)
    }

    pub fn get_mut_with<R>(&self, id: SlotId, f: impl FnOnce(&mut T) -> R) -> Option<R> {
        let mut arena = self.inner.write();
        arena.get_mut(id).map(f)
    }

    pub fn contains(&self, id: SlotId) -> bool {
        let arena = self.inner.read();
        arena.contains(id)
    }

    pub fn len(&self) -> usize {
        let arena = self.inner.read();
        arena.len()
    }

    pub fn is_empty(&self) -> bool {
        let arena = self.inner.read();
        arena.is_empty()
    }

    pub fn capacity(&self) -> usize {
        let arena = self.inner.read();
        arena.capacity()
    }

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
}
