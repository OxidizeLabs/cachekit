use crate::ds::{IntrusiveList, SlotArena, SlotId};
use crate::store::hashmap::HashMapStore;
use crate::store::traits::{StoreCore, StoreMut};
use std::collections::VecDeque;
use std::hash::Hash;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum QueueKind {
    Probation,
    Protected,
}

#[derive(Debug)]
struct Entry<K, V> {
    key: K,
    value: V,
    queue: QueueKind,
}

#[derive(Debug)]
pub struct LruQueue<T> {
    list: IntrusiveList<T>,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct TwoQWithGhost<K, V> {
    core: TwoQCore<K, V>,
    ghost_list: VecDeque<K>,
    ghost_list_cap: usize,
}

#[derive(Debug)]
pub struct TwoQCore<K, V> {
    index: HashMapStore<K, SlotId>, // key → store id
    store: SlotArena<Entry<K, V>>,

    probation: IntrusiveList<SlotId>, // FIFO probation
    protected: LruQueue<SlotId>,      // LRU protected

    probation_cap: usize,
    protected_cap: usize,
}

impl<T> Default for LruQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> LruQueue<T> {
    pub fn new() -> Self {
        Self {
            list: IntrusiveList::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.list.len() == 0
    }

    pub fn insert(&mut self, id: T) -> SlotId {
        // new item is most-recently-used
        self.list.push_front(id)
    }

    pub fn touch(&mut self, id: SlotId) -> bool {
        // move accessed item to MRU position
        self.list.move_to_front(id)
    }

    pub fn evict(&mut self) -> Option<T> {
        // remove least-recently-used
        self.list.pop_back()
    }

    pub fn remove(&mut self, id: SlotId) -> Option<T> {
        self.list.remove(id)
    }

    pub fn len(&self) -> usize {
        self.list.len()
    }

    pub fn push_front(&mut self, id: T) -> SlotId {
        self.list.push_front(id)
    }

    pub fn move_to_front(&mut self, id: SlotId) -> bool {
        self.list.move_to_front(id)
    }
    pub fn pop_back(&mut self) -> Option<T> {
        self.list.pop_back()
    }
}

impl<K, V> TwoQCore<K, V>
where
    K: Clone + Eq + Hash,
{
    pub fn new(protected_cap: usize, a1_frac: f64) -> Self {
        let probation_cap = (protected_cap as f64 * a1_frac) as usize;

        Self {
            index: HashMapStore::new(protected_cap),
            store: SlotArena::new(),
            probation: IntrusiveList::new(),
            protected: LruQueue::new(),
            probation_cap,
            protected_cap,
        }
    }

    pub fn get(&mut self, key: &K) -> Option<&V> {
        let id = self.index.get(key)?;

        let queue = self.store.get(*id)?.queue;
        match queue {
            QueueKind::Probation => {
                self.probation.remove(*id);
                self.probation.push_front(*id);
                if let Some(e) = self.store.get_mut(*id) {
                    e.queue = QueueKind::Protected;
                }
            },
            QueueKind::Protected => {
                self.protected.move_to_front(*id);
            },
        }

        self.store.get(*id).map(|e| &e.value)
    }

    pub fn insert(&mut self, key: K, value: V) {
        if let Some(id) = self.index.get(&key) {
            if let Some(e) = self.store.get_mut(*id) {
                e.value = value;
            }
            return;
        }

        let entry = Entry {
            key: key.clone(),
            value,
            queue: QueueKind::Probation,
        };
        let id = self.store.insert(entry);

        self.index
            .try_insert(key, id)
            .expect("Failed to insert entry");
        self.probation.push_back(id);

        self.evict_if_needed();
    }

    fn evict_if_needed(&mut self) {
        while self.len() > self.protected_cap {
            if self.probation.len() > self.probation_cap {
                if let Some(id) = self.probation.pop_front() {
                    self.remove_id(id);
                }
            } else if let Some(id) = self.protected.pop_back() {
                self.remove_id(id);
            }
        }
    }

    fn remove_id(&mut self, id: SlotId) {
        if let Some(entry) = self.store.remove(id) {
            self.index.remove(&entry.key);
        }
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.len() == 0
    }

    pub fn capacity(&self) -> usize {
        self.protected_cap
    }

    pub fn contains(&self, key: &K) -> bool {
        self.index.contains(key)
    }

    pub fn clear(&mut self) {
        self.index.clear();
        self.store.clear();
        self.probation.clear();
        self.protected.list.clear();
    }
}

impl<K, V> crate::traits::CoreCache<K, V> for TwoQCore<K, V>
where
    K: Clone + Eq + Hash,
{
    fn insert(&mut self, key: K, value: V) -> Option<V> {
        // Check if key exists - update in place
        if let Some(id) = self.index.get(&key) {
            if let Some(e) = self.store.get_mut(*id) {
                let old = std::mem::replace(&mut e.value, value);
                return Some(old);
            }
        }

        // New insert
        TwoQCore::insert(self, key, value);
        None
    }

    fn get(&mut self, key: &K) -> Option<&V> {
        TwoQCore::get(self, key)
    }

    fn contains(&self, key: &K) -> bool {
        self.index.contains(key)
    }

    fn len(&self) -> usize {
        self.index.len()
    }

    fn capacity(&self) -> usize {
        self.protected_cap
    }

    fn clear(&mut self) {
        self.index.clear();
        self.store.clear();
        self.probation.clear();
        self.protected.list.clear();
    }
}
