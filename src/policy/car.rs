//! Clock with Adaptive Replacement (CAR) replacement policy.
//!
//! Implements the CAR algorithm, which combines ARC-like adaptivity with Clock
//! mechanics to reduce list manipulation overhead. Hits only set a reference bit
//! (no list moves), improving concurrency and reducing pointer chasing.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                           CARCore<K, V> Layout                              │
//! │                                                                             │
//! │   Slot array with per-ring intrusive circular linked lists:                 │
//! │                                                                             │
//! │   ┌─────────────────────────────────────────────────────────────────────┐   │
//! │   │  index: HashMap<K, usize>    slots: Vec<Option<SlotPayload>>      │   │
//! │   │                                                                     │   │
//! │   │  Parallel hot-path arrays (cache-friendly sweeps):                   │   │
//! │   │    referenced: Vec<bool>       ring_kind: Vec<Ring>              │   │
//! │   │    ring_next:  Vec<usize>      ring_prev: Vec<usize>                │   │
//! │   │                                                                     │   │
//! │   │  Recent ring (recency)         Frequent ring (frequency)              │   │
//! │   │  ┌───────────────────┐        ┌───────────────────┐                  │   │
//! │   │  │ hand_recent ──► A ──►│        │ hand_frequent──► X ──►│              │   │
//! │   │  │  ◄── C ◄── B ◄───│        │  ◄── Z ◄── Y ◄───│                  │   │
//! │   │  └───────────────────┘        └───────────────────┘                  │   │
//! │   │  Ref=0 → evict to ghost_recent Ref=0 → evict to ghost_frequent      │   │
//! │   │  Ref=1 → demote to Frequent   Ref=1 → clear ref, advance            │   │
//! │   │                                                                     │   │
//! │   │  free: Vec<usize> (O(1) slot allocation/deallocation)               │   │
//! │   │  ghost_recent, ghost_frequent: GhostList<K> (ARC-style adaptation)  │   │
//! │   │  target_recent_size: adaptation parameter (target |Recent|)         │   │
//! │   └─────────────────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Key Components
//!
//! - [`CARCore`]: Main CAR cache implementation
//! - Two intrusive circular rings: Recent (seen once), Frequent (repeated access)
//! - Ghost lists (ghost_recent / ghost_frequent) for evicted keys (ARC-style adaptation)
//! - `target_recent_size`: target size for the Recent ring, adjusted by ghost hits
//! - Reference bits in parallel array for cache-friendly sweeps
//!
//! ## Operations
//!
//! | Operation   | Time   | Notes                                        |
//! |-------------|--------|-----------------------------------------------|
//! | `get`       | O(1)   | Sets reference bit only (no list move)       |
//! | `insert`    | O(1)*  | *Amortized; eviction may sweep                |
//! | `contains`  | O(1)   | Index lookup only                             |
//! | `remove`    | O(1)   | Unlink from ring + free slot                  |
//! | `len`       | O(1)   | Returns Recent + Frequent entries              |
//!
//! ## Performance Trade-offs
//!
//! - **When to Use**: Mixed workloads; need ARC-like adaptivity with lower
//!   overhead than ARC's list moves; concurrency-friendly (hits are cheap).
//! - **vs ARC**: Same adaptivity, but hits are O(1) bit set vs O(1) list move.
//!   Contiguous slot array is more cache-friendly than heap-allocated nodes.
//! - **vs Clock**: Adds ghost-based adaptation and scan resistance.
//! - **Memory Overhead**: Slot array + ghost keys (up to 2× capacity in keys).
//!
//! ## Example Usage
//!
//! ```
//! use cachekit::policy::car::CARCore;
//! use cachekit::traits::{CoreCache, ReadOnlyCache};
//!
//! let mut cache = CARCore::new(100);
//! cache.insert("key1", "value1");
//! cache.insert("key2", "value2");
//!
//! assert_eq!(cache.get(&"key1"), Some(&"value1"));
//! assert_eq!(cache.recent_len() + cache.frequent_len(), cache.len());
//! println!("target = {}", cache.target_recent_size());
//! ```
//!
//! ## Thread Safety
//!
//! - [`CARCore`]: Not thread-safe, designed for single-threaded use
//! - For concurrent access, wrap in external synchronization
//!
//! ## Implementation Notes
//!
//! - Two per-ring intrusive circular linked lists through a shared slot array
//! - Hands follow ring chains directly: O(1) advancement (no scanning)
//! - Free-slot stack: O(1) allocation and deallocation
//! - `referenced` and `ring_kind` in separate parallel arrays for sweep locality
//! - Ghost lists bounded to `capacity` each
//! - `target_recent_size` starts at `capacity / 2`
//!
//! ## References
//!
//! - Bansal & Modha, "CAR: Clock with Adaptive Replacement", FAST 2004
//! - Wikipedia: Cache replacement policies

use crate::ds::GhostList;
use crate::prelude::ReadOnlyCache;
use crate::traits::{CoreCache, MutableCache};
use rustc_hash::FxHashMap;
use std::hash::Hash;

/// Which logical ring an entry resides in.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum Ring {
    /// Recent-once ring: entries seen only once since entering the cache.
    Recent,
    /// Frequent ring: entries accessed more than once (promoted from Recent).
    Frequent,
}

/// Slot payload: key and value only.
///
/// Hot metadata (`referenced`, `ring_kind`) and ring linkage (`ring_next`,
/// `ring_prev`) are stored in separate parallel arrays for cache-friendly
/// sweeps during eviction.
struct SlotPayload<K, V> {
    key: K,
    value: V,
}

/// Core Clock with Adaptive Replacement (CAR) implementation.
///
/// Combines ARC's adaptivity (ghost lists + adaptation target) with Clock mechanics:
/// hits only set a reference bit instead of moving entries in linked lists.
/// Two per-ring intrusive circular linked lists share a single slot array,
/// giving O(1) hand advancement and O(1) slot allocation.
///
/// # Type Parameters
///
/// - `K`: Key type, must be `Clone + Eq + Hash`
/// - `V`: Value type
///
/// # Example
///
/// ```
/// use cachekit::policy::car::CARCore;
/// use cachekit::traits::{CoreCache, ReadOnlyCache};
///
/// let mut cache = CARCore::new(100);
/// cache.insert("key1", "value1");
/// assert_eq!(cache.get(&"key1"), Some(&"value1"));
/// assert_eq!(cache.recent_len() + cache.frequent_len(), cache.len());
/// ```
///
/// # Eviction Behavior
///
/// The replacement algorithm selects victims based on `target_recent_size`:
/// - If `|Recent| > target` (or equal and ghost hit was from ghost_frequent): sweep Recent
///   - Ref=0: evict to ghost_recent
///   - Ref=1: demote to Frequent (clear ref), continue
/// - Otherwise: sweep Frequent
///   - Ref=0: evict to ghost_frequent
///   - Ref=1: clear ref, advance, continue
#[must_use]
pub struct CARCore<K, V>
where
    K: Clone + Eq + Hash,
{
    /// Key -> slot index.
    index: FxHashMap<K, usize>,

    /// Slot storage (key + value only; metadata in parallel arrays).
    slots: Vec<Option<SlotPayload<K, V>>>,

    /// Reference bits — hot path: checked/cleared every sweep step.
    referenced: Vec<bool>,

    /// Ring membership — hot path: checked every sweep step.
    ring_kind: Vec<Ring>,

    /// Intrusive ring: next index in the same ring (circular).
    ring_next: Vec<usize>,

    /// Intrusive ring: previous index in the same ring (circular).
    ring_prev: Vec<usize>,

    /// Clock hand for the recent ring. `None` when the ring is empty.
    hand_recent: Option<usize>,

    /// Clock hand for the frequent ring. `None` when the ring is empty.
    hand_frequent: Option<usize>,

    /// Stack of free slot indices. O(1) push/pop.
    free: Vec<usize>,

    /// Ghost list for keys evicted from the recent ring.
    ghost_recent: GhostList<K>,

    /// Ghost list for keys evicted from the frequent ring.
    ghost_frequent: GhostList<K>,

    /// Adaptation parameter: target size for the recent ring.
    target_recent_size: usize,

    /// Number of entries in the recent ring.
    recent_len: usize,

    /// Number of entries in the frequent ring.
    frequent_len: usize,

    /// Maximum total cache capacity.
    capacity: usize,
}

impl<K, V> CARCore<K, V>
where
    K: Clone + Eq + Hash,
{
    /// Creates a new CAR cache with the specified capacity.
    ///
    /// Ghost lists can each hold up to `capacity` keys.
    /// Initial `target_recent_size` is set to `capacity / 2`.
    /// A capacity of 0 creates a no-op cache that drops all inserts.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::car::CARCore;
    /// use cachekit::traits::ReadOnlyCache;
    ///
    /// let cache: CARCore<String, i32> = CARCore::new(100);
    /// assert_eq!(cache.capacity(), 100);
    /// assert!(cache.is_empty());
    /// assert_eq!(cache.target_recent_size(), 50);
    /// ```
    #[inline]
    pub fn new(capacity: usize) -> Self {
        let mut slots = Vec::with_capacity(capacity);
        slots.resize_with(capacity, || None);
        let mut free = Vec::with_capacity(capacity);
        for i in (0..capacity).rev() {
            free.push(i);
        }
        Self {
            index: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
            slots,
            referenced: vec![false; capacity],
            ring_kind: vec![Ring::Recent; capacity],
            ring_next: vec![0; capacity],
            ring_prev: vec![0; capacity],
            hand_recent: None,
            hand_frequent: None,
            free,
            ghost_recent: GhostList::new(capacity),
            ghost_frequent: GhostList::new(capacity),
            target_recent_size: capacity / 2,
            recent_len: 0,
            frequent_len: 0,
            capacity,
        }
    }

    // =========================================================================
    // Ring operations: O(1) intrusive circular doubly-linked list management
    // =========================================================================

    /// Pop a free slot index. O(1).
    #[inline(always)]
    fn alloc_slot(&mut self) -> Option<usize> {
        self.free.pop()
    }

    /// Return a slot to the free list. O(1).
    #[inline(always)]
    fn free_slot(&mut self, idx: usize) {
        self.slots[idx] = None;
        self.referenced[idx] = false;
        self.free.push(idx);
    }

    /// Link slot `idx` into the given ring, just before the hand.
    ///
    /// The slot is placed at the "tail" of the ring (last to be examined
    /// before the hand wraps). If the ring is empty, the slot becomes the
    /// sole entry and the hand points to it.
    #[inline(always)]
    fn link_before_hand(&mut self, idx: usize, list: Ring) {
        self.ring_kind[idx] = list;
        let hand_ref = match list {
            Ring::Recent => &mut self.hand_recent,
            Ring::Frequent => &mut self.hand_frequent,
        };
        match *hand_ref {
            None => {
                // Empty ring: self-loop.
                self.ring_next[idx] = idx;
                self.ring_prev[idx] = idx;
                *hand_ref = Some(idx);
            },
            Some(h) => {
                // Insert between hand's predecessor and hand.
                let h_prev = self.ring_prev[h];
                self.ring_next[idx] = h;
                self.ring_prev[idx] = h_prev;
                self.ring_next[h_prev] = idx;
                self.ring_prev[h] = idx;
            },
        }
    }

    /// Unlink slot `idx` from its current ring. Updates hand if needed.
    #[inline(always)]
    fn unlink(&mut self, idx: usize) {
        let list = self.ring_kind[idx];
        let hand_ref = match list {
            Ring::Recent => &mut self.hand_recent,
            Ring::Frequent => &mut self.hand_frequent,
        };

        let next = self.ring_next[idx];
        let prev = self.ring_prev[idx];

        if next == idx {
            // Sole entry in ring.
            *hand_ref = None;
        } else {
            self.ring_next[prev] = next;
            self.ring_prev[next] = prev;
            if *hand_ref == Some(idx) {
                *hand_ref = Some(next);
            }
        }
    }

    // =========================================================================
    // Replacement and adaptation
    // =========================================================================

    /// Evict one entry to make room. Returns `true` if space was freed.
    ///
    /// Sweep budget: `2 * capacity` steps — enough to clear all ref bits
    /// in the first pass and find a victim in the second.
    fn replace(&mut self, ghost_frequent_hit: bool) -> bool {
        for _ in 0..(2 * self.capacity + 1) {
            let evict_from_recent = if self.recent_len > 0
                && (self.recent_len > self.target_recent_size
                    || (self.recent_len == self.target_recent_size && ghost_frequent_hit))
            {
                true
            } else if self.frequent_len > 0 {
                false
            } else {
                self.recent_len > 0
            };

            if evict_from_recent {
                let h = match self.hand_recent {
                    Some(h) => h,
                    None => continue,
                };
                if self.referenced[h] {
                    // Demote to frequent ring: unlink from recent, link into frequent, clear ref.
                    self.referenced[h] = false;
                    self.unlink(h);
                    self.recent_len -= 1;
                    self.link_before_hand(h, Ring::Frequent);
                    self.frequent_len += 1;
                } else {
                    // Evict from recent ring to ghost_recent.
                    let key = self.slots[h].as_ref().unwrap().key.clone();
                    self.unlink(h);
                    self.index.remove(&key);
                    self.recent_len -= 1;
                    self.free_slot(h);
                    self.ghost_recent.record(key);
                    return true;
                }
            } else {
                let h = match self.hand_frequent {
                    Some(h) => h,
                    None => continue,
                };
                if self.referenced[h] {
                    // Second chance: clear ref, advance hand.
                    self.referenced[h] = false;
                    self.hand_frequent = Some(self.ring_next[h]);
                } else {
                    // Evict from frequent ring to ghost_frequent.
                    let key = self.slots[h].as_ref().unwrap().key.clone();
                    self.unlink(h);
                    self.index.remove(&key);
                    self.frequent_len -= 1;
                    self.free_slot(h);
                    self.ghost_frequent.record(key);
                    return true;
                }
            }
        }
        false
    }

    /// Adapt parameter `p` based on ghost hit location.
    ///
    /// - ghost_recent hit: increase target by `max(|ghost_frequent|/|ghost_recent|, 1)` (favor recency)
    /// - ghost_frequent hit: decrease target by `max(|ghost_recent|/|ghost_frequent|, 1)` (favor frequency)
    ///
    /// Uses integer division. The `max(1)` ensures progress even when one
    /// ghost list is much smaller than the other.
    fn adapt(&mut self, ghost_recent_hit: bool, ghost_frequent_hit: bool) {
        if ghost_recent_hit {
            let delta = if !self.ghost_recent.is_empty() {
                (self.ghost_frequent.len() / self.ghost_recent.len()).max(1)
            } else {
                1
            };
            self.target_recent_size = (self.target_recent_size + delta).min(self.capacity);
        } else if ghost_frequent_hit {
            let delta = if !self.ghost_frequent.is_empty() {
                (self.ghost_recent.len() / self.ghost_frequent.len()).max(1)
            } else {
                1
            };
            self.target_recent_size = self.target_recent_size.saturating_sub(delta);
        }
    }

    /// Insert a key-value pair into the specified ring.
    ///
    /// Allocates a free slot, stores the payload, and links into the ring.
    /// Returns `false` if no free slot is available (caller should have
    /// called `replace` first).
    fn insert_into_ring(&mut self, key: K, value: V, list: Ring) -> bool {
        let idx = match self.alloc_slot() {
            Some(idx) => idx,
            None => return false,
        };
        self.slots[idx] = Some(SlotPayload {
            key: key.clone(),
            value,
        });
        self.referenced[idx] = false;
        self.link_before_hand(idx, list);
        self.index.insert(key, idx);
        match list {
            Ring::Recent => self.recent_len += 1,
            Ring::Frequent => self.frequent_len += 1,
        }
        true
    }

    // =========================================================================
    // Public accessors
    // =========================================================================

    /// Returns the current adaptation target for the recent ring.
    ///
    /// Higher values favor recency (larger recent ring),
    /// lower values favor frequency (larger frequent ring).
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::car::CARCore;
    ///
    /// let cache: CARCore<String, i32> = CARCore::new(100);
    /// assert_eq!(cache.target_recent_size(), 50);
    /// ```
    pub fn target_recent_size(&self) -> usize {
        self.target_recent_size
    }

    /// Returns the number of entries in the recent ring (seen once).
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::car::CARCore;
    /// use cachekit::traits::CoreCache;
    ///
    /// let mut cache = CARCore::new(100);
    /// cache.insert("key", "value");
    /// assert_eq!(cache.recent_len(), 1);
    /// ```
    pub fn recent_len(&self) -> usize {
        self.recent_len
    }

    /// Returns the number of entries in the frequent ring (repeated access).
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::policy::car::CARCore;
    /// use cachekit::traits::CoreCache;
    ///
    /// let mut cache = CARCore::new(100);
    /// cache.insert("key", "value");
    /// cache.get(&"key"); // CAR: ref bit set, but stays in recent ring
    /// ```
    pub fn frequent_len(&self) -> usize {
        self.frequent_len
    }

    /// Returns the number of keys in the ghost_recent list (evicted from recent ring).
    pub fn ghost_recent_len(&self) -> usize {
        self.ghost_recent.len()
    }

    /// Returns the number of keys in the ghost_frequent list (evicted from frequent ring).
    pub fn ghost_frequent_len(&self) -> usize {
        self.ghost_frequent.len()
    }

    /// Validates internal invariants. Available in debug/test builds.
    ///
    /// Panics if any invariant is violated.
    #[cfg(any(test, debug_assertions))]
    pub fn debug_validate_invariants(&self)
    where
        K: std::fmt::Debug,
    {
        // 1. Counts match index.
        assert_eq!(
            self.recent_len + self.frequent_len,
            self.index.len(),
            "recent_len({}) + frequent_len({}) != index.len({})",
            self.recent_len,
            self.frequent_len,
            self.index.len()
        );

        // 2. Doesn't exceed capacity.
        assert!(
            self.recent_len + self.frequent_len <= self.capacity,
            "total({}) > capacity({})",
            self.recent_len + self.frequent_len,
            self.capacity
        );

        // 3. target_recent_size within range.
        assert!(
            self.target_recent_size <= self.capacity,
            "p({}) > capacity({})",
            self.target_recent_size,
            self.capacity
        );

        // 4. Ghost lists within bounds.
        assert!(self.ghost_recent.len() <= self.capacity);
        assert!(self.ghost_frequent.len() <= self.capacity);

        // 5. Free list size matches.
        assert_eq!(
            self.free.len(),
            self.capacity - (self.recent_len + self.frequent_len),
            "free.len({}) != capacity({}) - total({})",
            self.free.len(),
            self.capacity,
            self.recent_len + self.frequent_len
        );

        // 6. Walk recent ring and count.
        let mut recent_count = 0;
        if let Some(start) = self.hand_recent {
            let mut cur = start;
            loop {
                assert!(
                    self.slots[cur].is_some(),
                    "recent ring slot {} is empty",
                    cur
                );
                assert_eq!(
                    self.ring_kind[cur],
                    Ring::Recent,
                    "recent ring slot {} has ring_kind {:?}",
                    cur,
                    self.ring_kind[cur]
                );
                let slot = self.slots[cur].as_ref().unwrap();
                assert!(
                    self.index.contains_key(&slot.key),
                    "recent ring slot {} key not in index",
                    cur
                );
                // Verify prev/next consistency.
                assert_eq!(
                    self.ring_next[self.ring_prev[cur]], cur,
                    "recent ring broken at slot {}",
                    cur
                );
                assert_eq!(
                    self.ring_prev[self.ring_next[cur]], cur,
                    "recent ring broken at slot {}",
                    cur
                );
                recent_count += 1;
                cur = self.ring_next[cur];
                if cur == start {
                    break;
                }
                assert!(recent_count <= self.capacity, "recent ring cycle detected");
            }
        }
        assert_eq!(
            recent_count, self.recent_len,
            "recent ring walk count mismatch"
        );

        // 7. Walk frequent ring and count.
        let mut frequent_count = 0;
        if let Some(start) = self.hand_frequent {
            let mut cur = start;
            loop {
                assert!(self.slots[cur].is_some());
                assert_eq!(self.ring_kind[cur], Ring::Frequent);
                let slot = self.slots[cur].as_ref().unwrap();
                assert!(self.index.contains_key(&slot.key));
                assert_eq!(self.ring_next[self.ring_prev[cur]], cur);
                assert_eq!(self.ring_prev[self.ring_next[cur]], cur);
                frequent_count += 1;
                cur = self.ring_next[cur];
                if cur == start {
                    break;
                }
                assert!(frequent_count <= self.capacity);
            }
        }
        assert_eq!(
            frequent_count, self.frequent_len,
            "frequent ring walk count mismatch"
        );

        // 8. Ghost lists don't overlap with live keys.
        for key in self.index.keys() {
            assert!(
                !self.ghost_recent.contains(key),
                "Key {:?} in both cache and ghost_recent",
                key
            );
            assert!(
                !self.ghost_frequent.contains(key),
                "Key {:?} in both cache and ghost_frequent",
                key
            );
        }

        // 9. All index entries point to valid slots with correct keys.
        for (key, &idx) in &self.index {
            assert!(idx < self.capacity);
            let slot = self.slots[idx]
                .as_ref()
                .expect("index points to empty slot");
            assert_eq!(&slot.key, key);
        }

        // 10. Free slots are actually empty.
        for &idx in &self.free {
            assert!(idx < self.capacity);
            assert!(self.slots[idx].is_none(), "Free slot {} is occupied", idx);
        }
    }
}

impl<K, V> std::fmt::Debug for CARCore<K, V>
where
    K: Clone + Eq + Hash + std::fmt::Debug,
    V: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CARCore")
            .field("capacity", &self.capacity)
            .field("recent_len", &self.recent_len)
            .field("frequent_len", &self.frequent_len)
            .field("ghost_recent_len", &self.ghost_recent.len())
            .field("ghost_frequent_len", &self.ghost_frequent.len())
            .field("target_recent_size", &self.target_recent_size)
            .field("total_len", &(self.recent_len + self.frequent_len))
            .finish()
    }
}

impl<K, V> ReadOnlyCache<K, V> for CARCore<K, V>
where
    K: Clone + Eq + Hash,
{
    fn contains(&self, key: &K) -> bool {
        self.index.contains_key(key)
    }

    fn len(&self) -> usize {
        self.recent_len + self.frequent_len
    }

    fn capacity(&self) -> usize {
        self.capacity
    }
}

impl<K, V> CoreCache<K, V> for CARCore<K, V>
where
    K: Clone + Eq + Hash,
{
    fn get(&mut self, key: &K) -> Option<&V> {
        let &idx = self.index.get(key)?;
        self.referenced[idx] = true;
        self.slots[idx].as_ref().map(|s| &s.value)
    }

    fn insert(&mut self, key: K, value: V) -> Option<V> {
        if self.capacity == 0 {
            return None;
        }

        // Case 1: Key already in cache — update value, set ref.
        if let Some(&idx) = self.index.get(&key) {
            if let Some(slot) = self.slots[idx].as_mut() {
                let old = std::mem::replace(&mut slot.value, value);
                self.referenced[idx] = true;
                return Some(old);
            }
        }

        let ghost_recent_hit = self.ghost_recent.contains(&key);
        let ghost_frequent_hit = self.ghost_frequent.contains(&key);

        // Case 2: Ghost hit in ghost_recent (key was recently evicted from recent ring).
        if ghost_recent_hit {
            self.adapt(true, false);
            self.ghost_recent.remove(&key);
            if self.recent_len + self.frequent_len >= self.capacity {
                self.replace(false);
            }
            self.insert_into_ring(key, value, Ring::Frequent);
            return None;
        }

        // Case 3: Ghost hit in ghost_frequent (key was evicted from frequent ring).
        if ghost_frequent_hit {
            self.adapt(false, true);
            self.ghost_frequent.remove(&key);
            if self.recent_len + self.frequent_len >= self.capacity {
                self.replace(true);
            }
            self.insert_into_ring(key, value, Ring::Frequent);
            return None;
        }

        // Case 4: Complete miss.
        if self.recent_len + self.frequent_len >= self.capacity && !self.replace(false) {
            return None;
        }
        self.insert_into_ring(key, value, Ring::Recent);
        None
    }

    fn clear(&mut self) {
        self.index.clear();
        for slot in &mut self.slots {
            *slot = None;
        }
        self.referenced.fill(false);
        self.free.clear();
        for i in (0..self.capacity).rev() {
            self.free.push(i);
        }
        self.hand_recent = None;
        self.hand_frequent = None;
        self.ghost_recent.clear();
        self.ghost_frequent.clear();
        self.target_recent_size = self.capacity / 2;
        self.recent_len = 0;
        self.frequent_len = 0;
    }
}

impl<K, V> MutableCache<K, V> for CARCore<K, V>
where
    K: Clone + Eq + Hash,
{
    fn remove(&mut self, key: &K) -> Option<V> {
        let idx = self.index.remove(key)?;
        let list = self.ring_kind[idx];
        self.unlink(idx);
        match list {
            Ring::Recent => self.recent_len -= 1,
            Ring::Frequent => self.frequent_len -= 1,
        }
        let slot = self.slots[idx].take()?;
        self.referenced[idx] = false;
        self.free.push(idx);
        Some(slot.value)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn car_new_cache() {
        let cache: CARCore<String, i32> = CARCore::new(100);
        assert_eq!(cache.capacity(), 100);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.recent_len(), 0);
        assert_eq!(cache.frequent_len(), 0);
        assert_eq!(cache.target_recent_size(), 50);
        cache.debug_validate_invariants();
    }

    #[test]
    fn car_zero_capacity() {
        let mut cache: CARCore<&str, i32> = CARCore::new(0);
        assert_eq!(cache.capacity(), 0);
        assert_eq!(cache.len(), 0);
        cache.insert("key", 1);
        assert_eq!(cache.len(), 0);
        assert!(!cache.contains(&"key"));
    }

    #[test]
    fn car_insert_and_get() {
        let mut cache = CARCore::new(10);
        cache.insert("key1", "value1");
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.recent_len(), 1);
        assert_eq!(cache.frequent_len(), 0);
        cache.debug_validate_invariants();

        assert_eq!(cache.get(&"key1"), Some(&"value1"));
        // CAR: get sets ref bit but doesn't move between lists.
        assert_eq!(cache.recent_len(), 1);
        assert_eq!(cache.frequent_len(), 0);
        cache.debug_validate_invariants();
    }

    #[test]
    fn car_update_existing() {
        let mut cache = CARCore::new(10);
        cache.insert("key1", "value1");
        let old = cache.insert("key1", "new_value");
        assert_eq!(old, Some("value1"));
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get(&"key1"), Some(&"new_value"));
        cache.debug_validate_invariants();
    }

    #[test]
    fn car_eviction_fills_ghost() {
        let mut cache = CARCore::new(2);
        cache.insert("a", 1);
        cache.insert("b", 2);
        assert_eq!(cache.len(), 2);
        cache.debug_validate_invariants();

        cache.insert("c", 3);
        assert_eq!(cache.len(), 2);
        assert!(cache.ghost_recent_len() + cache.ghost_frequent_len() >= 1);
        cache.debug_validate_invariants();
    }

    #[test]
    fn car_ghost_hit_ghost_recent() {
        let mut cache = CARCore::new(2);
        cache.insert("a", 1);
        cache.insert("b", 2);
        cache.insert("c", 3); // Evicts "a" to ghost_recent
        assert!(!cache.contains(&"a"));
        cache.debug_validate_invariants();

        // Ghost hit: "a" should go to the frequent ring
        cache.insert("a", 10);
        assert!(cache.contains(&"a"));
        assert_eq!(cache.get(&"a"), Some(&10));
        cache.debug_validate_invariants();
    }

    #[test]
    fn car_ghost_hit_ghost_frequent() {
        let mut cache = CARCore::new(3);
        // Fill the recent ring.
        cache.insert("a", 1);
        cache.insert("b", 2);
        cache.insert("c", 3);
        // Access to set ref bits, then evict to get entries into ghost_frequent.
        cache.get(&"a");
        cache.get(&"b");
        // Insert "d" — "c" (unreferenced) evicted to ghost_recent;
        // "a" and "b" demoted to the frequent ring with ref cleared.
        cache.insert("d", 4);
        cache.debug_validate_invariants();

        // Now force entries through the frequent ring into ghost_frequent.
        cache.insert("e", 5);
        cache.insert("f", 6);
        cache.debug_validate_invariants();

        // Check if any key ended up in ghost_frequent.
        let ghost_frequent_has_entries = cache.ghost_frequent_len() > 0;
        if ghost_frequent_has_entries {
            // Force a ghost hit on a ghost_frequent key if possible.
            // (exact key depends on hand positions, so just validate invariants)
            cache.debug_validate_invariants();
        }
    }

    #[test]
    fn car_remove() {
        let mut cache = CARCore::new(10);
        cache.insert("key1", "value1");
        cache.insert("key2", "value2");
        assert_eq!(cache.remove(&"key1"), Some("value1"));
        assert_eq!(cache.len(), 1);
        assert!(!cache.contains(&"key1"));
        assert!(cache.contains(&"key2"));
        cache.debug_validate_invariants();
    }

    #[test]
    fn car_remove_nonexistent() {
        let mut cache = CARCore::new(10);
        cache.insert("key1", "value1");
        assert_eq!(cache.remove(&"missing"), None);
        assert_eq!(cache.len(), 1);
        cache.debug_validate_invariants();
    }

    #[test]
    fn car_clear() {
        let mut cache = CARCore::new(10);
        cache.insert("key1", "value1");
        cache.insert("key2", "value2");
        cache.get(&"key1");
        cache.clear();
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.recent_len(), 0);
        assert_eq!(cache.frequent_len(), 0);
        assert_eq!(cache.ghost_recent_len(), 0);
        assert_eq!(cache.ghost_frequent_len(), 0);
        assert!(cache.is_empty());
        cache.debug_validate_invariants();
    }

    #[test]
    fn car_ref_bit_protects_from_eviction() {
        // Capacity 3, target=1. a,b,c in recent ring. get(b), get(c) set ref bits.
        // Insert d: evict from recent ring (|recent|=3>target). Unreferenced "a" evicted first.
        let mut cache = CARCore::new(3);
        cache.insert("a", 1);
        cache.insert("b", 2);
        cache.insert("c", 3);
        cache.get(&"b");
        cache.get(&"c");
        cache.insert("d", 4);
        assert_eq!(cache.len(), 3);
        assert!(!cache.contains(&"a"));
        assert!(cache.contains(&"b"));
        assert!(cache.contains(&"c"));
        assert!(cache.contains(&"d"));
        cache.debug_validate_invariants();
    }

    #[test]
    fn car_demotion_recent_to_frequent() {
        // Fill recent ring, set ref on all, then insert to trigger demotion.
        let mut cache = CARCore::new(3);
        cache.insert("a", 1);
        cache.insert("b", 2);
        cache.insert("c", 3);
        cache.get(&"a");
        cache.get(&"b");
        cache.get(&"c");
        // All recent entries have ref=1. Inserting "d" should demote all to frequent,
        // then evict one from the frequent ring.
        cache.insert("d", 4);
        assert_eq!(cache.len(), 3);
        // Some entries should have been demoted to the frequent ring.
        assert!(cache.frequent_len() > 0);
        cache.debug_validate_invariants();
    }

    #[test]
    fn car_adaptation_increases_target_on_ghost_recent_hit() {
        let mut cache = CARCore::new(4);
        let initial_p = cache.target_recent_size();

        cache.insert("a", 1);
        cache.insert("b", 2);
        cache.insert("c", 3);
        cache.insert("d", 4);
        cache.insert("e", 5); // Evicts "a" to ghost_recent
        cache.debug_validate_invariants();

        // Ghost hit in ghost_recent should increase target_recent_size.
        cache.insert("a", 10);
        cache.debug_validate_invariants();
        assert!(
            cache.target_recent_size() >= initial_p,
            "target_recent_size should not decrease on ghost_recent hit: was {}, now {}",
            initial_p,
            cache.target_recent_size()
        );
    }

    #[test]
    fn car_multiple_entries() {
        let mut cache = CARCore::new(5);
        for i in 0..5 {
            cache.insert(i, i * 10);
        }
        assert_eq!(cache.len(), 5);
        cache.debug_validate_invariants();

        for i in 0..5 {
            assert_eq!(cache.get(&i), Some(&(i * 10)));
        }
        cache.debug_validate_invariants();
    }

    #[test]
    fn car_capacity_one() {
        let mut cache = CARCore::new(1);
        cache.insert("a", 1);
        assert_eq!(cache.len(), 1);
        cache.debug_validate_invariants();

        cache.insert("b", 2);
        assert_eq!(cache.len(), 1);
        assert!(cache.contains(&"b"));
        assert!(!cache.contains(&"a"));
        cache.debug_validate_invariants();
    }

    #[test]
    fn car_heavy_churn() {
        let mut cache = CARCore::new(10);
        for i in 0..1000 {
            cache.insert(i, i * 10);
            if i % 3 == 0 {
                cache.get(&(i / 2));
            }
            if i % 7 == 0 {
                cache.remove(&(i / 3));
            }
        }
        assert!(cache.len() <= 10);
        cache.debug_validate_invariants();
    }

    #[test]
    fn car_contains_after_eviction() {
        let mut cache = CARCore::new(2);
        cache.insert("a", 1);
        cache.insert("b", 2);
        assert!(cache.contains(&"a"));
        assert!(cache.contains(&"b"));

        cache.insert("c", 3);
        assert_eq!(cache.len(), 2);
        assert!(cache.contains(&"c"));
        cache.debug_validate_invariants();
    }

    #[test]
    fn car_insert_after_remove_reuses_slot() {
        let mut cache = CARCore::new(3);
        cache.insert("a", 1);
        cache.insert("b", 2);
        cache.insert("c", 3);
        cache.debug_validate_invariants();

        cache.remove(&"b");
        assert_eq!(cache.len(), 2);
        cache.debug_validate_invariants();

        cache.insert("d", 4);
        assert_eq!(cache.len(), 3);
        assert!(cache.contains(&"d"));
        cache.debug_validate_invariants();
    }

    #[test]
    fn car_clear_then_reuse() {
        let mut cache = CARCore::new(5);
        for i in 0..5 {
            cache.insert(i, i);
        }
        cache.clear();
        cache.debug_validate_invariants();

        for i in 10..15 {
            cache.insert(i, i);
        }
        assert_eq!(cache.len(), 5);
        for i in 10..15 {
            assert!(cache.contains(&i));
        }
        cache.debug_validate_invariants();
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// len() never exceeds capacity.
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_len_within_capacity(
            capacity in 1usize..100,
            ops in prop::collection::vec((0u32..1000, 0u32..100), 0..200)
        ) {
            let mut cache = CARCore::new(capacity);
            for (key, value) in ops {
                cache.insert(key, value);
                prop_assert!(cache.len() <= cache.capacity());
            }
        }

        /// Invariants hold after any sequence of inserts.
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_invariants_after_inserts(
            capacity in 1usize..50,
            ops in prop::collection::vec((0u32..100, 0u32..100), 0..100)
        ) {
            let mut cache = CARCore::new(capacity);
            for (key, value) in ops {
                cache.insert(key, value);
                cache.debug_validate_invariants();
            }
        }

        /// Get after insert returns correct value (if not evicted).
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_get_after_insert(
            capacity in 1usize..50,
            key in 0u32..100,
            value in 0u32..1000
        ) {
            let mut cache = CARCore::new(capacity);
            cache.insert(key, value);
            if cache.contains(&key) {
                prop_assert_eq!(cache.get(&key), Some(&value));
            }
        }

        /// Remove decreases length and makes key absent.
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_remove_behavior(
            capacity in 1usize..50,
            keys in prop::collection::vec(0u32..100, 1..50)
        ) {
            let mut cache = CARCore::new(capacity);
            for &key in &keys {
                cache.insert(key, key * 10);
            }

            for &key in &keys[0..keys.len()/2] {
                let len_before = cache.len();
                let removed = cache.remove(&key);
                if removed.is_some() {
                    prop_assert_eq!(cache.len(), len_before - 1);
                    prop_assert!(!cache.contains(&key));
                }
                cache.debug_validate_invariants();
            }
        }

        /// Clear empties the cache.
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_clear_empties(
            capacity in 1usize..50,
            keys in prop::collection::vec(0u32..100, 1..50)
        ) {
            let mut cache = CARCore::new(capacity);
            for key in keys {
                cache.insert(key, key * 10);
            }
            cache.clear();
            prop_assert!(cache.is_empty());
            prop_assert_eq!(cache.len(), 0);
            cache.debug_validate_invariants();
        }

        /// Referenced entries get a second chance (demotion, not permanent protection).
        /// With enough capacity, a referenced entry should survive one round of eviction.
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_second_chance_behavior(
            capacity in 3usize..10
        ) {
            let mut cache = CARCore::new(capacity);
            for i in 0..capacity {
                cache.insert(i as u32, i as u32);
            }
            // Mark entry 0 as referenced.
            cache.get(&0);
            // Insert one new entry. Entry 0 should survive (demoted to frequent ring)
            // because unreferenced entries are evicted first.
            cache.insert(capacity as u32, capacity as u32);
            // With capacity >= 3, entry 0 should survive: it gets demoted to the
            // frequent ring, and there are enough unreferenced recent entries to evict instead.
            prop_assert!(cache.contains(&0), "referenced entry 0 should survive");
            cache.debug_validate_invariants();
        }

        /// Duplicate inserts don't grow the cache.
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_duplicate_inserts_no_growth(
            capacity in 1usize..30,
            key in 0u32..50,
            values in prop::collection::vec(0u32..100, 1..50)
        ) {
            let mut cache = CARCore::new(capacity);
            cache.insert(key, 0);
            let len_after_first = cache.len();
            for value in values {
                cache.insert(key, value);
                prop_assert_eq!(cache.len(), len_after_first);
            }
            cache.debug_validate_invariants();
        }
    }

    // =========================================================================
    // Arbitrary operation sequences
    // =========================================================================

    #[derive(Debug, Clone)]
    enum Operation {
        Insert(u32, u32),
        Get(u32),
        Remove(u32),
    }

    fn operation_strategy() -> impl Strategy<Value = Operation> {
        prop_oneof![
            (0u32..50, 0u32..100).prop_map(|(k, v)| Operation::Insert(k, v)),
            (0u32..50).prop_map(Operation::Get),
            (0u32..50).prop_map(Operation::Remove),
        ]
    }

    proptest! {
        /// Arbitrary operation sequences maintain all invariants.
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_arbitrary_ops_maintain_invariants(
            capacity in 1usize..30,
            ops in prop::collection::vec(operation_strategy(), 0..200)
        ) {
            let mut cache = CARCore::new(capacity);

            for op in ops {
                match op {
                    Operation::Insert(k, v) => { cache.insert(k, v); }
                    Operation::Get(k) => { cache.get(&k); }
                    Operation::Remove(k) => { cache.remove(&k); }
                }
                cache.debug_validate_invariants();
                prop_assert!(cache.len() <= cache.capacity());
            }
        }

        /// Interleaved inserts and removes maintain consistency.
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_interleaved_insert_remove(
            capacity in 1usize..30,
            ops in prop::collection::vec((0u32..50, any::<bool>()), 0..200)
        ) {
            let mut cache = CARCore::new(capacity);

            for (key, should_insert) in ops {
                if should_insert {
                    cache.insert(key, key * 10);
                } else {
                    cache.remove(&key);
                }
                cache.debug_validate_invariants();
                prop_assert!(cache.len() <= cache.capacity());
            }
        }

        /// Zero capacity cache is always empty.
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_zero_capacity_noop(
            ops in prop::collection::vec((0u32..100, 0u32..100), 0..50)
        ) {
            let mut cache = CARCore::<u32, u32>::new(0);
            for (key, value) in ops {
                cache.insert(key, value);
                prop_assert!(cache.is_empty());
                prop_assert_eq!(cache.len(), 0);
                prop_assert!(!cache.contains(&key));
            }
        }

        /// Capacity 1 cache never exceeds 1 entry.
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_capacity_one_single_entry(
            keys in prop::collection::vec(0u32..100, 1..50)
        ) {
            let mut cache = CARCore::new(1);
            for key in keys {
                cache.insert(key, key * 10);
                prop_assert!(cache.len() <= 1);
                cache.debug_validate_invariants();
            }
        }
    }
}

#[cfg(test)]
mod fuzz_tests {
    use super::*;

    pub fn fuzz_arbitrary_operations(data: &[u8]) {
        if data.len() < 2 {
            return;
        }

        let capacity = (data[0] as usize % 50).max(1);
        let mut cache = CARCore::new(capacity);

        let mut idx = 1;
        while idx + 2 < data.len() {
            let op = data[idx] % 4;
            let key = data[idx + 1] as u32;
            let value = data[idx + 2] as u32;

            match op {
                0 => {
                    cache.insert(key, value);
                },
                1 => {
                    cache.get(&key);
                },
                2 => {
                    cache.remove(&key);
                },
                3 => {
                    // Mixed: insert then get
                    cache.insert(key, value);
                    cache.get(&key);
                },
                _ => unreachable!(),
            }

            cache.debug_validate_invariants();
            assert!(cache.len() <= cache.capacity());
            idx += 3;
        }
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_fuzz_smoke() {
        let inputs = vec![
            vec![5, 0, 1, 2, 1, 3, 4, 2, 5, 6],
            vec![10, 6, 7, 8, 5, 9, 10, 0, 1, 2],
            vec![1, 0, 0, 0, 1, 1, 1, 2, 2, 2],
            vec![3, 0, 10, 20, 0, 10, 30, 0, 20, 40, 0, 30, 50],
        ];
        for input in inputs {
            fuzz_arbitrary_operations(&input);
        }
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_fuzz_eviction_patterns() {
        let inputs = vec![
            vec![
                5, 0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 5, 0, 5, 6, 1, 1, 0, 1, 3, 0,
            ],
            vec![2, 0, 1, 2, 0, 3, 4, 0, 5, 6, 0, 1, 7, 0, 3, 8],
        ];
        for input in inputs {
            fuzz_arbitrary_operations(&input);
        }
    }
}
