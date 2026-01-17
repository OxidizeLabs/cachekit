//! Slot arena with stable [`SlotId`] handles.
//!
//! Provides arena-style allocation where elements are stored in a `Vec<Option<T>>`
//! and freed slots are tracked in a free list for reuse. Handles ([`SlotId`]) remain
//! stable across insertions and deletions, making this ideal for graph structures,
//! linked lists, and policy metadata.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                           SlotArena Layout                                  │
//! │                                                                             │
//! │   ┌───────────────────────────────────────────────────────────────────┐    │
//! │   │  slots: Vec<Option<T>>                                            │    │
//! │   │                                                                   │    │
//! │   │    index:   0        1        2        3        4        5       │    │
//! │   │           ┌────┐  ┌────┐  ┌────┐  ┌────┐  ┌────┐  ┌────┐        │    │
//! │   │           │ T  │  │None│  │ T  │  │None│  │ T  │  │None│        │    │
//! │   │           │"a" │  │    │  │"c" │  │    │  │"e" │  │    │        │    │
//! │   │           └────┘  └────┘  └────┘  └────┘  └────┘  └────┘        │    │
//! │   │              ▲       │       ▲       │       ▲       │           │    │
//! │   │              │       │       │       │       │       │           │    │
//! │   │           SlotId(0)  │   SlotId(2)   │   SlotId(4)   │           │    │
//! │   │                      │               │               │           │    │
//! │   └──────────────────────┼───────────────┼───────────────┼───────────┘    │
//! │                          │               │               │                 │
//! │   ┌──────────────────────┼───────────────┼───────────────┼───────────┐    │
//! │   │  free_list: Vec<usize>               │               │           │    │
//! │   │                      │               │               │           │    │
//! │   │              ┌───────┴───────┬───────┴───────┬───────┴───────┐  │    │
//! │   │              │       1       │       3       │       5       │  │    │
//! │   │              └───────────────┴───────────────┴───────────────┘  │    │
//! │   │                                                                  │    │
//! │   │              ◄─────── pop() returns 5 for next insert ──────►   │    │
//! │   └──────────────────────────────────────────────────────────────────┘    │
//! │                                                                             │
//! │   len: 3  (number of live entries, not slots.len())                        │
//! └─────────────────────────────────────────────────────────────────────────────┘
//!
//! Slot Lifecycle
//! ──────────────
//!
//!   insert("x"):
//!     ├─► free_list.pop() ──► Some(idx) ──► slots[idx] = Some("x")
//!     │                                      return SlotId(idx)
//!     │
//!     └─► None ──► slots.push(Some("x"))
//!                  return SlotId(slots.len() - 1)
//!
//!   remove(SlotId(2)):
//!     slots[2].take() ──► Some("c") ──► free_list.push(2)
//!                                       return Some("c")
//!
//!   get(SlotId(2)):
//!     slots.get(2) ──► Some(&slot) ──► slot.as_ref() ──► Option<&T>
//!
//! Arena Variants
//! ──────────────
//!
//!   ┌─────────────────────────────────────────────────────────────────────┐
//!   │ SlotArena<T>              Single-threaded, direct &T access         │
//!   ├─────────────────────────────────────────────────────────────────────┤
//!   │ ConcurrentSlotArena<T>    Thread-safe via RwLock                    │
//!   │                           Uses closures: get_with(id, |v| ...)      │
//!   ├─────────────────────────────────────────────────────────────────────┤
//!   │ ShardedSlotArena<T>       Multiple RwLock-protected arenas          │
//!   │                           Round-robin insert, ShardedSlotId handle  │
//!   └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Key Components
//!
//! - [`SlotId`]: Stable handle wrapping a `usize` index
//! - [`SlotArena`]: Single-threaded arena with O(1) operations
//! - [`ConcurrentSlotArena`]: Thread-safe wrapper with closure-based access
//! - [`ShardedSlotArena`]: Sharded arena for high-concurrency workloads
//! - [`ShardedSlotId`]: Handle containing shard index + slot id
//!
//! ## Operations
//!
//! | Operation     | Description                              | Complexity |
//! |---------------|------------------------------------------|------------|
//! | `insert`      | Add value, reuse free slot if available  | O(1)       |
//! | `remove`      | Remove and return value, free the slot   | O(1)       |
//! | `get`         | Get reference by SlotId                  | O(1)       |
//! | `get_mut`     | Get mutable reference by SlotId          | O(1)       |
//! | `contains`    | Check if SlotId is valid                 | O(1)       |
//! | `iter`        | Iterate over live entries                | O(n)       |
//!
//! ## Performance Characteristics
//!
//! - **No per-operation allocation**: Slots are reused, Vec grows as needed
//! - **Cache-friendly**: Elements stored contiguously in memory
//! - **Stable handles**: SlotId values remain valid until removed
//! - **O(1) operations**: All single-element operations are constant time
//!
//! ## When to Use
//!
//! - Intrusive data structures (linked lists, trees, graphs)
//! - Policy metadata storage (LRU list nodes, frequency counters)
//! - Any scenario requiring stable handles that survive mutations
//! - Memory pools where allocation churn is a concern
//!
//! ## Example Usage
//!
//! ```
//! use cachekit::ds::SlotArena;
//!
//! let mut arena = SlotArena::new();
//!
//! // Insert values and get stable handles
//! let id_a = arena.insert("alice");
//! let id_b = arena.insert("bob");
//!
//! // Access by handle
//! assert_eq!(arena.get(id_a), Some(&"alice"));
//!
//! // Remove frees the slot
//! assert_eq!(arena.remove(id_b), Some("bob"));
//!
//! // Next insert reuses freed slot
//! let id_c = arena.insert("carol");
//! assert_eq!(id_b.index(), id_c.index());  // Same underlying index
//! ```
//!
//! ## Thread Safety
//!
//! - [`SlotArena`]: Not thread-safe, use in single-threaded contexts
//! - [`ConcurrentSlotArena`]: Thread-safe via `parking_lot::RwLock`
//! - [`ShardedSlotArena`]: Thread-safe with per-shard locks
//!
//! ## Implementation Notes
//!
//! - `SlotId` indices may be reused after removal
//! - Accessing a stale `SlotId` returns `None` (not undefined behavior)
//! - `len()` tracks live entries, not `slots.len()`
//! - `debug_validate_invariants()` available in debug/test builds
/// Stable handle into a [`SlotArena`].
///
/// A lightweight identifier (wrapping a `usize` index) that provides O(1)
/// access to arena slots. Handles remain valid until the slot is removed;
/// after removal, the index may be reused by a later `insert`.
///
/// # Stability
///
/// Unlike raw indices or pointers, `SlotId` values are semantically tied to
/// the slot they reference. Accessing a removed slot returns `None` rather
/// than causing undefined behavior.
///
/// # Example
///
/// ```
/// use cachekit::ds::SlotArena;
///
/// let mut arena = SlotArena::new();
/// let id = arena.insert(42);
///
/// // SlotId provides O(1) access
/// assert_eq!(arena.get(id), Some(&42));
///
/// // Inspect raw index for debugging
/// println!("Value stored at index {}", id.index());
///
/// // After removal, same index may be reused
/// arena.remove(id);
/// let new_id = arena.insert(100);
/// assert_eq!(id.index(), new_id.index());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SlotId(pub(crate) usize);

impl SlotId {
    /// Returns the underlying slot index.
    ///
    /// Useful for debugging, logging, or custom data structures that need
    /// to work with raw indices.
    pub fn index(self) -> usize {
        self.0
    }
}

/// Single-threaded arena with stable [`SlotId`] handles and O(1) operations.
///
/// Stores values in a `Vec<Option<T>>` and reuses freed slots via a free list.
/// Ideal for graph structures, linked lists, and policy metadata where stable
/// handles are needed.
///
/// # Example
///
/// ```
/// use cachekit::ds::SlotArena;
///
/// let mut arena = SlotArena::new();
///
/// // Insert returns stable handles
/// let a = arena.insert("first");
/// let b = arena.insert("second");
/// let c = arena.insert("third");
///
/// assert_eq!(arena.len(), 3);
/// assert_eq!(arena.get(b), Some(&"second"));
///
/// // Remove frees the slot
/// arena.remove(b);
/// assert_eq!(arena.len(), 2);
/// assert_eq!(arena.get(b), None);
///
/// // Next insert reuses the freed slot
/// let d = arena.insert("fourth");
/// assert_eq!(b.index(), d.index());
///
/// // Iterate over live entries
/// for (id, value) in arena.iter() {
///     println!("{}: {}", id.index(), value);
/// }
/// ```
///
/// # Use Case: Intrusive Linked List
///
/// ```
/// use cachekit::ds::{SlotArena, SlotId};
///
/// struct Node {
///     value: i32,
///     next: Option<SlotId>,
/// }
///
/// let mut arena = SlotArena::new();
///
/// // Build a linked list: 1 -> 2 -> 3
/// let c = arena.insert(Node { value: 3, next: None });
/// let b = arena.insert(Node { value: 2, next: Some(c) });
/// let a = arena.insert(Node { value: 1, next: Some(b) });
///
/// // Traverse
/// let mut current = Some(a);
/// while let Some(id) = current {
///     let node = arena.get(id).unwrap();
///     println!("{}", node.value);
///     current = node.next;
/// }
/// ```
#[derive(Debug)]
pub struct SlotArena<T> {
    slots: Vec<Option<T>>,
    free_list: Vec<usize>,
    len: usize,
}

impl<T> SlotArena<T> {
    /// Creates an empty arena.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::SlotArena;
    ///
    /// let arena: SlotArena<String> = SlotArena::new();
    /// assert!(arena.is_empty());
    /// ```
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_list: Vec::new(),
            len: 0,
        }
    }

    /// Creates an empty arena with pre-allocated capacity.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::SlotArena;
    ///
    /// let arena: SlotArena<i32> = SlotArena::with_capacity(100);
    /// assert!(arena.capacity() >= 100);
    /// assert!(arena.is_empty());
    /// ```
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            slots: Vec::with_capacity(capacity),
            free_list: Vec::new(),
            len: 0,
        }
    }

    /// Inserts a value and returns its [`SlotId`].
    ///
    /// Reuses a freed slot if available, otherwise grows the internal Vec.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::SlotArena;
    ///
    /// let mut arena = SlotArena::new();
    /// let id = arena.insert("hello");
    /// assert_eq!(arena.get(id), Some(&"hello"));
    /// ```
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

    /// Removes and returns the value at `id`, freeing the slot for reuse.
    ///
    /// Returns `None` if the slot is empty or out of bounds.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::SlotArena;
    ///
    /// let mut arena = SlotArena::new();
    /// let id = arena.insert(42);
    ///
    /// assert_eq!(arena.remove(id), Some(42));
    /// assert_eq!(arena.remove(id), None);  // Already removed
    /// ```
    pub fn remove(&mut self, id: SlotId) -> Option<T> {
        let slot = self.slots.get_mut(id.0)?;
        let value = slot.take()?;
        self.free_list.push(id.0);
        self.len -= 1;
        Some(value)
    }

    /// Returns a shared reference to the value at `id`.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::SlotArena;
    ///
    /// let mut arena = SlotArena::new();
    /// let id = arena.insert("value");
    ///
    /// assert_eq!(arena.get(id), Some(&"value"));
    /// ```
    pub fn get(&self, id: SlotId) -> Option<&T> {
        self.slots.get(id.0).and_then(|slot| slot.as_ref())
    }

    /// Returns a mutable reference to the value at `id`.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::SlotArena;
    ///
    /// let mut arena = SlotArena::new();
    /// let id = arena.insert(1);
    ///
    /// if let Some(v) = arena.get_mut(id) {
    ///     *v = 2;
    /// }
    /// assert_eq!(arena.get(id), Some(&2));
    /// ```
    pub fn get_mut(&mut self, id: SlotId) -> Option<&mut T> {
        self.slots.get_mut(id.0).and_then(|slot| slot.as_mut())
    }

    /// Returns `true` if `id` refers to a live slot.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::SlotArena;
    ///
    /// let mut arena = SlotArena::new();
    /// let id = arena.insert(1);
    ///
    /// assert!(arena.contains(id));
    /// arena.remove(id);
    /// assert!(!arena.contains(id));
    /// ```
    pub fn contains(&self, id: SlotId) -> bool {
        self.slots
            .get(id.0)
            .map(|slot| slot.is_some())
            .unwrap_or(false)
    }

    /// Returns `true` if the raw `index` is in bounds and occupied.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::SlotArena;
    ///
    /// let mut arena = SlotArena::new();
    /// let id = arena.insert("value");
    ///
    /// assert!(arena.contains_index(id.index()));
    /// assert!(!arena.contains_index(999));
    /// ```
    pub fn contains_index(&self, index: usize) -> bool {
        self.slots
            .get(index)
            .map(|slot| slot.is_some())
            .unwrap_or(false)
    }

    /// Returns the number of live entries.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::SlotArena;
    ///
    /// let mut arena = SlotArena::new();
    /// assert_eq!(arena.len(), 0);
    ///
    /// arena.insert("a");
    /// arena.insert("b");
    /// assert_eq!(arena.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the arena has no live entries.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::SlotArena;
    ///
    /// let mut arena = SlotArena::new();
    /// assert!(arena.is_empty());
    ///
    /// let id = arena.insert(1);
    /// assert!(!arena.is_empty());
    ///
    /// arena.remove(id);
    /// assert!(arena.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the backing Vec capacity (slots that fit without reallocation).
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::SlotArena;
    ///
    /// let arena: SlotArena<i32> = SlotArena::with_capacity(100);
    /// assert!(arena.capacity() >= 100);
    /// ```
    pub fn capacity(&self) -> usize {
        self.slots.capacity()
    }

    /// Reserves capacity for at least `additional` more slots.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::SlotArena;
    ///
    /// let mut arena: SlotArena<i32> = SlotArena::new();
    /// arena.reserve_slots(100);
    /// assert!(arena.capacity() >= 100);
    /// ```
    pub fn reserve_slots(&mut self, additional: usize) {
        self.slots.reserve(additional);
    }

    /// Shrinks internal storage to fit the current state.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::SlotArena;
    ///
    /// let mut arena: SlotArena<i32> = SlotArena::with_capacity(1000);
    /// arena.insert(1);
    /// arena.shrink_to_fit();
    /// // Capacity reduced (exact value is implementation-defined)
    /// ```
    pub fn shrink_to_fit(&mut self) {
        self.slots.shrink_to_fit();
        self.free_list.shrink_to_fit();
    }

    /// Clears all entries and shrinks internal storage.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::SlotArena;
    ///
    /// let mut arena = SlotArena::with_capacity(100);
    /// arena.insert("a");
    /// arena.insert("b");
    ///
    /// arena.clear_shrink();
    /// assert!(arena.is_empty());
    /// ```
    pub fn clear_shrink(&mut self) {
        self.clear();
        self.shrink_to_fit();
    }

    /// Removes all entries and resets internal state.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::SlotArena;
    ///
    /// let mut arena = SlotArena::new();
    /// let a = arena.insert("a");
    /// let b = arena.insert("b");
    ///
    /// arena.clear();
    /// assert!(arena.is_empty());
    /// assert!(!arena.contains(a));
    /// assert!(!arena.contains(b));
    /// ```
    pub fn clear(&mut self) {
        self.slots.clear();
        self.free_list.clear();
        self.len = 0;
    }

    /// Iterates over live `(SlotId, &T)` pairs.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::SlotArena;
    ///
    /// let mut arena = SlotArena::new();
    /// arena.insert("a");
    /// arena.insert("b");
    ///
    /// let values: Vec<_> = arena.iter().map(|(_, v)| *v).collect();
    /// assert_eq!(values, vec!["a", "b"]);
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = (SlotId, &T)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(idx, slot)| slot.as_ref().map(|value| (SlotId(idx), value)))
    }

    /// Iterates over live [`SlotId`]s only.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::SlotArena;
    ///
    /// let mut arena = SlotArena::new();
    /// let a = arena.insert("a");
    /// let b = arena.insert("b");
    /// arena.remove(a);
    ///
    /// let ids: Vec<_> = arena.iter_ids().collect();
    /// assert_eq!(ids.len(), 1);
    /// assert!(ids.contains(&b));
    /// ```
    pub fn iter_ids(&self) -> impl Iterator<Item = SlotId> + '_ {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(idx, slot)| slot.as_ref().map(|_| SlotId(idx)))
    }

    #[cfg(any(test, debug_assertions))]
    /// Returns a debug snapshot of arena internals.
    pub fn debug_snapshot(&self) -> SlotArenaSnapshot {
        let mut free_list = self.free_list.clone();
        free_list.sort_unstable();
        let mut live_ids: Vec<_> = self.iter_ids().collect();
        live_ids.sort_by_key(|id| id.index());
        SlotArenaSnapshot {
            len: self.len,
            slots_len: self.slots.len(),
            free_list,
            live_ids,
        }
    }

    /// Returns an approximate memory footprint in bytes.
    ///
    /// Includes the struct itself, slot storage, and free list.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::SlotArena;
    ///
    /// let mut arena: SlotArena<u64> = SlotArena::with_capacity(100);
    /// let bytes = arena.approx_bytes();
    /// assert!(bytes > 0);
    /// ```
    pub fn approx_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.slots.capacity() * std::mem::size_of::<Option<T>>()
            + self.free_list.capacity() * std::mem::size_of::<usize>()
    }

    #[cfg(any(test, debug_assertions))]
    /// Validates internal invariants (debug/test builds only).
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

#[cfg(any(test, debug_assertions))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotArenaSnapshot {
    pub len: usize,
    pub slots_len: usize,
    pub free_list: Vec<usize>,
    pub live_ids: Vec<SlotId>,
}

/// Stable handle into a [`ShardedSlotArena`].
///
/// Contains both the shard index and the [`SlotId`] within that shard.
/// Required for O(1) access since the shard must be known to locate the value.
///
/// # Example
///
/// ```
/// use cachekit::ds::ShardedSlotArena;
///
/// let arena = ShardedSlotArena::new(4);
/// let id = arena.insert(42);
///
/// // Inspect shard and slot
/// println!("Shard {}, Slot {}", id.shard(), id.slot().index());
///
/// // Access value
/// assert_eq!(arena.get_with(id, |v| *v), Some(42));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShardedSlotId {
    shard: usize,
    slot: SlotId,
}

impl ShardedSlotId {
    /// Returns the shard index.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ShardedSlotArena;
    ///
    /// let arena = ShardedSlotArena::new(4);
    /// let id = arena.insert(42);
    /// assert!(id.shard() < 4);
    /// ```
    pub fn shard(self) -> usize {
        self.shard
    }

    /// Returns the [`SlotId`] within the shard.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ShardedSlotArena;
    ///
    /// let arena = ShardedSlotArena::new(2);
    /// let id = arena.insert("value");
    /// let slot = id.slot();
    /// // slot.index() is the position within the shard
    /// ```
    pub fn slot(self) -> SlotId {
        self.slot
    }
}

/// Thread-safe sharded arena with multiple `RwLock`-protected [`SlotArena`]s.
///
/// Distributes inserts across shards in round-robin fashion to reduce
/// contention. Each shard has its own lock, so operations on different
/// shards can proceed in parallel.
///
/// # Use Cases
///
/// - High-concurrency metadata storage
/// - When a single `ConcurrentSlotArena` becomes a bottleneck
/// - Workloads with many concurrent writers
///
/// # Example
///
/// ```
/// use std::sync::Arc;
/// use std::thread;
/// use cachekit::ds::ShardedSlotArena;
///
/// let arena = Arc::new(ShardedSlotArena::new(4));
///
/// // Spawn writers
/// let handles: Vec<_> = (0..4).map(|_| {
///     let arena = Arc::clone(&arena);
///     thread::spawn(move || {
///         for i in 0..100 {
///             arena.insert(i);
///         }
///     })
/// }).collect();
///
/// for h in handles {
///     h.join().unwrap();
/// }
///
/// assert_eq!(arena.len(), 400);
/// assert_eq!(arena.shard_count(), 4);
/// ```
#[derive(Debug)]
pub struct ShardedSlotArena<T> {
    shards: Vec<parking_lot::RwLock<SlotArena<T>>>,
    selector: crate::ds::ShardSelector,
    next_shard: std::sync::atomic::AtomicUsize,
}

impl<T> ShardedSlotArena<T> {
    /// Creates a sharded arena with the specified number of shards.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ShardedSlotArena;
    ///
    /// let arena: ShardedSlotArena<i32> = ShardedSlotArena::new(8);
    /// assert_eq!(arena.shard_count(), 8);
    /// ```
    pub fn new(shards: usize) -> Self {
        let shards = shards.max(1);
        let mut arenas = Vec::with_capacity(shards);
        for _ in 0..shards {
            arenas.push(parking_lot::RwLock::new(SlotArena::new()));
        }
        Self {
            shards: arenas,
            selector: crate::ds::ShardSelector::new(shards, 0),
            next_shard: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Creates a sharded arena with pre-allocated capacity per shard.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ShardedSlotArena;
    ///
    /// // 4 shards, each pre-allocated for 1000 entries
    /// let arena: ShardedSlotArena<String> = ShardedSlotArena::with_capacity(4, 1000);
    /// assert_eq!(arena.shard_count(), 4);
    /// ```
    pub fn with_capacity(shards: usize, capacity: usize) -> Self {
        let shards = shards.max(1);
        let mut arenas = Vec::with_capacity(shards);
        for _ in 0..shards {
            arenas.push(parking_lot::RwLock::new(SlotArena::with_capacity(capacity)));
        }
        Self {
            shards: arenas,
            selector: crate::ds::ShardSelector::new(shards, 0),
            next_shard: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Alias for [`with_capacity`](Self::with_capacity).
    pub fn with_shards(shards: usize, capacity_per_shard: usize) -> Self {
        Self::with_capacity(shards, capacity_per_shard)
    }

    /// Creates a sharded arena with a custom hash seed for shard selection.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ShardedSlotArena;
    ///
    /// // Custom seed for deterministic shard selection
    /// let arena: ShardedSlotArena<i32> = ShardedSlotArena::with_shards_seed(4, 100, 42);
    /// assert_eq!(arena.shard_count(), 4);
    /// ```
    pub fn with_shards_seed(shards: usize, capacity_per_shard: usize, seed: u64) -> Self {
        let shards = shards.max(1);
        let mut arenas = Vec::with_capacity(shards);
        for _ in 0..shards {
            arenas.push(parking_lot::RwLock::new(SlotArena::with_capacity(
                capacity_per_shard,
            )));
        }
        Self {
            shards: arenas,
            selector: crate::ds::ShardSelector::new(shards, seed),
            next_shard: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Returns the shard index for a key using the configured hash selector.
    ///
    /// Useful for co-locating related data in the same shard.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ShardedSlotArena;
    ///
    /// let arena: ShardedSlotArena<String> = ShardedSlotArena::new(8);
    /// let shard = arena.shard_for_key(&"my_key");
    /// assert!(shard < 8);
    /// ```
    pub fn shard_for_key<K: std::hash::Hash>(&self, key: &K) -> usize {
        self.selector.shard_for_key(key)
    }

    /// Returns the number of shards.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ShardedSlotArena;
    ///
    /// let arena: ShardedSlotArena<i32> = ShardedSlotArena::new(16);
    /// assert_eq!(arena.shard_count(), 16);
    /// ```
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// Inserts a value into the next shard (round-robin) and returns its handle.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ShardedSlotArena;
    ///
    /// let arena = ShardedSlotArena::new(2);
    /// let a = arena.insert("first");
    /// let b = arena.insert("second");
    ///
    /// // Inserts alternate between shards
    /// assert_ne!(a.shard(), b.shard());
    /// ```
    pub fn insert(&self, value: T) -> ShardedSlotId {
        let shard = self
            .next_shard
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            % self.shards.len();
        let mut arena = self.shards[shard].write();
        let slot = arena.insert(value);
        ShardedSlotId { shard, slot }
    }

    /// Removes and returns the value at `id`.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ShardedSlotArena;
    ///
    /// let arena = ShardedSlotArena::new(2);
    /// let id = arena.insert(42);
    ///
    /// assert_eq!(arena.remove(id), Some(42));
    /// assert_eq!(arena.remove(id), None);  // Already removed
    /// ```
    pub fn remove(&self, id: ShardedSlotId) -> Option<T> {
        let mut arena = self.shards.get(id.shard)?.write();
        arena.remove(id.slot)
    }

    /// Runs a closure on a shared reference to the value at `id`.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ShardedSlotArena;
    ///
    /// let arena = ShardedSlotArena::new(2);
    /// let id = arena.insert(vec![1, 2, 3]);
    ///
    /// let len = arena.get_with(id, |v| v.len());
    /// assert_eq!(len, Some(3));
    /// ```
    pub fn get_with<R>(&self, id: ShardedSlotId, f: impl FnOnce(&T) -> R) -> Option<R> {
        let arena = self.shards.get(id.shard)?.read();
        arena.get(id.slot).map(f)
    }

    /// Runs a closure on a mutable reference to the value at `id`.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ShardedSlotArena;
    ///
    /// let arena = ShardedSlotArena::new(2);
    /// let id = arena.insert(10);
    ///
    /// arena.get_mut_with(id, |v| *v += 5);
    /// assert_eq!(arena.get_with(id, |v| *v), Some(15));
    /// ```
    pub fn get_mut_with<R>(&self, id: ShardedSlotId, f: impl FnOnce(&mut T) -> R) -> Option<R> {
        let mut arena = self.shards.get(id.shard)?.write();
        arena.get_mut(id.slot).map(f)
    }

    /// Returns `true` if `id` refers to a live slot.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ShardedSlotArena;
    ///
    /// let arena = ShardedSlotArena::new(2);
    /// let id = arena.insert("value");
    ///
    /// assert!(arena.contains(id));
    /// arena.remove(id);
    /// assert!(!arena.contains(id));
    /// ```
    pub fn contains(&self, id: ShardedSlotId) -> bool {
        self.shards
            .get(id.shard)
            .map(|arena| arena.read().contains(id.slot))
            .unwrap_or(false)
    }

    /// Returns the total number of live entries across all shards.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ShardedSlotArena;
    ///
    /// let arena = ShardedSlotArena::new(4);
    /// arena.insert(1);
    /// arena.insert(2);
    /// arena.insert(3);
    ///
    /// assert_eq!(arena.len(), 3);
    /// ```
    pub fn len(&self) -> usize {
        self.shards.iter().map(|arena| arena.read().len()).sum()
    }

    /// Returns `true` if all shards are empty.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ShardedSlotArena;
    ///
    /// let arena: ShardedSlotArena<i32> = ShardedSlotArena::new(2);
    /// assert!(arena.is_empty());
    ///
    /// arena.insert(1);
    /// assert!(!arena.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Reserves capacity in each shard.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ShardedSlotArena;
    ///
    /// let arena: ShardedSlotArena<i32> = ShardedSlotArena::new(4);
    /// arena.reserve_slots(100);  // Each shard gets 100 additional capacity
    /// ```
    pub fn reserve_slots(&self, additional: usize) {
        for arena in &self.shards {
            arena.write().reserve_slots(additional);
        }
    }

    /// Shrinks all shards to fit their current state.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ShardedSlotArena;
    ///
    /// let arena: ShardedSlotArena<i32> = ShardedSlotArena::with_capacity(4, 1000);
    /// arena.insert(1);
    /// arena.shrink_to_fit();
    /// ```
    pub fn shrink_to_fit(&self) {
        for arena in &self.shards {
            arena.write().shrink_to_fit();
        }
    }

    /// Clears all shards and shrinks storage.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ShardedSlotArena;
    ///
    /// let arena = ShardedSlotArena::new(2);
    /// arena.insert(1);
    /// arena.insert(2);
    ///
    /// arena.clear_shrink();
    /// assert!(arena.is_empty());
    /// ```
    pub fn clear_shrink(&self) {
        for arena in &self.shards {
            arena.write().clear_shrink();
        }
    }

    /// Returns approximate memory footprint across all shards.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ShardedSlotArena;
    ///
    /// let arena: ShardedSlotArena<u64> = ShardedSlotArena::with_capacity(4, 100);
    /// let bytes = arena.approx_bytes();
    /// assert!(bytes > 0);
    /// ```
    pub fn approx_bytes(&self) -> usize {
        self.shards
            .iter()
            .map(|arena| arena.read().approx_bytes())
            .sum()
    }
}

impl<T> Default for SlotArena<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-safe [`SlotArena`] wrapper using `parking_lot::RwLock`.
///
/// Provides the same functionality as [`SlotArena`] but safe for concurrent
/// access. Uses closure-based access (`get_with`, `get_mut_with`) since
/// references cannot outlive lock guards.
///
/// For high-contention workloads, consider [`ShardedSlotArena`] instead.
///
/// # Example
///
/// ```
/// use std::sync::Arc;
/// use std::thread;
/// use cachekit::ds::ConcurrentSlotArena;
///
/// let arena = Arc::new(ConcurrentSlotArena::new());
///
/// // Insert from main thread
/// let id = arena.insert(0);
///
/// // Spawn readers and writers
/// let handles: Vec<_> = (0..4).map(|i| {
///     let arena = Arc::clone(&arena);
///     thread::spawn(move || {
///         // Read
///         let val = arena.get_with(id, |v| *v);
///
///         // Write (increment)
///         arena.get_mut_with(id, |v| *v += 1);
///     })
/// }).collect();
///
/// for h in handles {
///     h.join().unwrap();
/// }
///
/// assert_eq!(arena.get_with(id, |v| *v), Some(4));
/// ```
#[derive(Debug)]
pub struct ConcurrentSlotArena<T> {
    inner: parking_lot::RwLock<SlotArena<T>>,
}

impl<T> ConcurrentSlotArena<T> {
    /// Creates an empty concurrent arena.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ConcurrentSlotArena;
    ///
    /// let arena: ConcurrentSlotArena<String> = ConcurrentSlotArena::new();
    /// assert!(arena.is_empty());
    /// ```
    pub fn new() -> Self {
        Self {
            inner: parking_lot::RwLock::new(SlotArena::new()),
        }
    }

    /// Creates a concurrent arena with pre-allocated capacity.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ConcurrentSlotArena;
    ///
    /// let arena: ConcurrentSlotArena<i32> = ConcurrentSlotArena::with_capacity(1000);
    /// assert!(arena.capacity() >= 1000);
    /// ```
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: parking_lot::RwLock::new(SlotArena::with_capacity(capacity)),
        }
    }

    /// Inserts a value and returns its [`SlotId`].
    ///
    /// Acquires write lock.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ConcurrentSlotArena;
    ///
    /// let arena = ConcurrentSlotArena::new();
    /// let id = arena.insert(42);
    /// assert!(arena.contains(id));
    /// ```
    pub fn insert(&self, value: T) -> SlotId {
        let mut arena = self.inner.write();
        arena.insert(value)
    }

    /// Removes and returns the value at `id`.
    ///
    /// Acquires write lock.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ConcurrentSlotArena;
    ///
    /// let arena = ConcurrentSlotArena::new();
    /// let id = arena.insert(42);
    ///
    /// assert_eq!(arena.remove(id), Some(42));
    /// assert_eq!(arena.remove(id), None);
    /// ```
    pub fn remove(&self, id: SlotId) -> Option<T> {
        let mut arena = self.inner.write();
        arena.remove(id)
    }

    /// Runs a closure on a shared reference to the value at `id`.
    ///
    /// Acquires read lock. The closure's return value is returned.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ConcurrentSlotArena;
    ///
    /// let arena = ConcurrentSlotArena::new();
    /// let id = arena.insert(vec![1, 2, 3]);
    ///
    /// let sum = arena.get_with(id, |v| v.iter().sum::<i32>());
    /// assert_eq!(sum, Some(6));
    /// ```
    pub fn get_with<R>(&self, id: SlotId, f: impl FnOnce(&T) -> R) -> Option<R> {
        let arena = self.inner.read();
        arena.get(id).map(f)
    }

    /// Non-blocking version of [`get_with`](Self::get_with).
    ///
    /// Returns `None` if the lock cannot be acquired immediately.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ConcurrentSlotArena;
    ///
    /// let arena = ConcurrentSlotArena::new();
    /// let id = arena.insert(100);
    ///
    /// // Returns Some if lock is available
    /// if let Some(val) = arena.try_get_with(id, |v| *v) {
    ///     assert_eq!(val, 100);
    /// }
    /// ```
    pub fn try_get_with<R>(&self, id: SlotId, f: impl FnOnce(&T) -> R) -> Option<R> {
        let arena = self.inner.try_read()?;
        arena.get(id).map(f)
    }

    /// Runs a closure on a mutable reference to the value at `id`.
    ///
    /// Acquires write lock.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ConcurrentSlotArena;
    ///
    /// let arena = ConcurrentSlotArena::new();
    /// let id = arena.insert(1);
    ///
    /// arena.get_mut_with(id, |v| *v = 2);
    /// assert_eq!(arena.get_with(id, |v| *v), Some(2));
    /// ```
    pub fn get_mut_with<R>(&self, id: SlotId, f: impl FnOnce(&mut T) -> R) -> Option<R> {
        let mut arena = self.inner.write();
        arena.get_mut(id).map(f)
    }

    /// Non-blocking version of [`get_mut_with`](Self::get_mut_with).
    ///
    /// Returns `None` if the lock cannot be acquired immediately.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ConcurrentSlotArena;
    ///
    /// let arena = ConcurrentSlotArena::new();
    /// let id = arena.insert(5);
    ///
    /// // Attempt non-blocking write
    /// if arena.try_get_mut_with(id, |v| *v += 1).is_some() {
    ///     assert_eq!(arena.get_with(id, |v| *v), Some(6));
    /// }
    /// ```
    pub fn try_get_mut_with<R>(&self, id: SlotId, f: impl FnOnce(&mut T) -> R) -> Option<R> {
        let mut arena = self.inner.try_write()?;
        arena.get_mut(id).map(f)
    }

    /// Returns `true` if `id` refers to a live slot.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ConcurrentSlotArena;
    ///
    /// let arena = ConcurrentSlotArena::new();
    /// let id = arena.insert("value");
    ///
    /// assert!(arena.contains(id));
    /// arena.remove(id);
    /// assert!(!arena.contains(id));
    /// ```
    pub fn contains(&self, id: SlotId) -> bool {
        let arena = self.inner.read();
        arena.contains(id)
    }

    /// Returns the number of live entries.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ConcurrentSlotArena;
    ///
    /// let arena = ConcurrentSlotArena::new();
    /// assert_eq!(arena.len(), 0);
    ///
    /// arena.insert(1);
    /// arena.insert(2);
    /// assert_eq!(arena.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        let arena = self.inner.read();
        arena.len()
    }

    /// Returns `true` if there are no live entries.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ConcurrentSlotArena;
    ///
    /// let arena: ConcurrentSlotArena<i32> = ConcurrentSlotArena::new();
    /// assert!(arena.is_empty());
    ///
    /// let id = arena.insert(1);
    /// assert!(!arena.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        let arena = self.inner.read();
        arena.is_empty()
    }

    /// Returns the backing Vec capacity.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ConcurrentSlotArena;
    ///
    /// let arena: ConcurrentSlotArena<i32> = ConcurrentSlotArena::with_capacity(100);
    /// assert!(arena.capacity() >= 100);
    /// ```
    pub fn capacity(&self) -> usize {
        let arena = self.inner.read();
        arena.capacity()
    }

    /// Reserves capacity for additional slots.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ConcurrentSlotArena;
    ///
    /// let arena: ConcurrentSlotArena<i32> = ConcurrentSlotArena::new();
    /// arena.reserve_slots(100);
    /// assert!(arena.capacity() >= 100);
    /// ```
    pub fn reserve_slots(&self, additional: usize) {
        let mut arena = self.inner.write();
        arena.reserve_slots(additional);
    }

    /// Shrinks internal storage to fit current state.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ConcurrentSlotArena;
    ///
    /// let arena: ConcurrentSlotArena<i32> = ConcurrentSlotArena::with_capacity(1000);
    /// arena.insert(1);
    /// arena.shrink_to_fit();
    /// ```
    pub fn shrink_to_fit(&self) {
        let mut arena = self.inner.write();
        arena.shrink_to_fit();
    }

    /// Clears all entries and shrinks storage.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ConcurrentSlotArena;
    ///
    /// let arena = ConcurrentSlotArena::with_capacity(100);
    /// arena.insert("a");
    /// arena.insert("b");
    ///
    /// arena.clear_shrink();
    /// assert!(arena.is_empty());
    /// ```
    pub fn clear_shrink(&self) {
        let mut arena = self.inner.write();
        arena.clear_shrink();
    }

    /// Clears all entries.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ConcurrentSlotArena;
    ///
    /// let arena = ConcurrentSlotArena::new();
    /// let a = arena.insert(1);
    /// let b = arena.insert(2);
    ///
    /// arena.clear();
    /// assert!(arena.is_empty());
    /// assert!(!arena.contains(a));
    /// ```
    pub fn clear(&self) {
        let mut arena = self.inner.write();
        arena.clear();
    }

    /// Returns approximate memory footprint in bytes.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::ConcurrentSlotArena;
    ///
    /// let arena: ConcurrentSlotArena<u64> = ConcurrentSlotArena::with_capacity(100);
    /// let bytes = arena.approx_bytes();
    /// assert!(bytes > 0);
    /// ```
    pub fn approx_bytes(&self) -> usize {
        let arena = self.inner.read();
        arena.approx_bytes()
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
    fn slot_arena_contains_index_and_iter_ids() {
        let mut arena = SlotArena::new();
        let a = arena.insert("a");
        let b = arena.insert("b");
        assert!(arena.contains_index(a.index()));
        assert!(arena.contains_index(b.index()));
        arena.remove(a);
        assert!(!arena.contains_index(a.index()));

        let ids: Vec<_> = arena.iter_ids().collect();
        assert_eq!(ids, vec![b]);
    }

    #[test]
    fn slot_arena_reserve_slots_grows_capacity() {
        let mut arena: SlotArena<i32> = SlotArena::new();
        let before = arena.capacity();
        arena.reserve_slots(32);
        assert!(arena.capacity() >= before + 32);
    }

    #[test]
    fn slot_arena_debug_snapshot() {
        let mut arena = SlotArena::new();
        let a = arena.insert("a");
        let b = arena.insert("b");
        arena.remove(a);
        let snapshot = arena.debug_snapshot();
        assert_eq!(snapshot.len, 1);
        assert!(snapshot.live_ids.contains(&b));
        assert!(snapshot.free_list.contains(&a.index()));
    }

    #[test]
    fn sharded_slot_arena_basic_ops() {
        let arena = ShardedSlotArena::new(2);
        let a = arena.insert(1);
        let b = arena.insert(2);
        assert!(arena.contains(a));
        assert!(arena.contains(b));
        assert_eq!(arena.get_with(a, |v| *v), Some(1));
        assert_eq!(arena.remove(b), Some(2));
        assert!(!arena.contains(b));
        assert_eq!(arena.len(), 1);
    }

    #[test]
    fn sharded_slot_arena_shard_for_key() {
        let arena: ShardedSlotArena<i32> = ShardedSlotArena::new(4);
        let shard = arena.shard_for_key(&"alpha");
        assert!(shard < arena.shard_count());
    }

    #[test]
    fn sharded_slot_arena_with_seed() {
        let arena: ShardedSlotArena<i32> = ShardedSlotArena::with_shards_seed(4, 0, 99);
        let shard = arena.shard_for_key(&"alpha");
        assert!(shard < arena.shard_count());
    }

    #[test]
    fn concurrent_slot_arena_try_ops() {
        let arena = ConcurrentSlotArena::new();
        let id = arena.insert(1);
        assert_eq!(arena.try_get_with(id, |v| *v), Some(1));
        arena.try_get_mut_with(id, |v| *v = 2);
        assert_eq!(arena.get_with(id, |v| *v), Some(2));
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
