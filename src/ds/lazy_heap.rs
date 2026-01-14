//! Lazy min-heap with stale entry skipping.
//!
//! Maintains a `BinaryHeap` of `(score, seq)` plus a `scores` map for the
//! latest value. Updates push new heap entries; `pop_best` skips stale ones.
//!
//! ## Architecture
//!
//! ```text
//!   scores (authoritative)
//!   ┌─────────┬──────┐
//!   │  key A  │  10  │
//!   │  key B  │   3  │
//!   └─────────┴──────┘
//!
//!   heap (may contain stale entries)
//!   min: (B,3,seq=5), (A,10,seq=2), (A,12,seq=1 stale)
//! ```
//!
//! ## Operations
//! - `update(k, s)`: updates map and pushes a heap entry
//! - `pop_best()`: pops until top matches current score
//! - `rebuild()`: rebuilds heap from authoritative map
//!
//! ## Performance
//! - `update` / `remove`: O(1) average
//! - `pop_best`: amortized O(log n)
//! - `rebuild`: O(n) when heap grows too stale
//!
//! `debug_validate_invariants()` is available in debug/test builds.
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};
use std::hash::Hash;

#[derive(Debug)]
struct HeapEntry<K, S> {
    score: S,
    seq: u64,
    key: K,
}

impl<K, S> PartialEq for HeapEntry<K, S>
where
    S: Ord,
{
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.seq == other.seq
    }
}

impl<K, S> Eq for HeapEntry<K, S> where S: Ord {}

impl<K, S> PartialOrd for HeapEntry<K, S>
where
    S: Ord,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<K, S> Ord for HeapEntry<K, S>
where
    S: Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        match self.score.cmp(&other.score) {
            Ordering::Equal => self.seq.cmp(&other.seq),
            ordering => ordering,
        }
    }
}

#[derive(Debug)]
/// Min-heap that supports cheap updates via lazy deletion.
pub struct LazyMinHeap<K, S> {
    scores: HashMap<K, S>,
    heap: BinaryHeap<Reverse<HeapEntry<K, S>>>,
    seq: u64,
}

impl<K, S> LazyMinHeap<K, S>
where
    K: Eq + Hash + Clone,
    S: Ord + Clone,
{
    /// Creates an empty heap.
    pub fn new() -> Self {
        Self {
            scores: HashMap::new(),
            heap: BinaryHeap::new(),
            seq: 0,
        }
    }

    /// Creates an empty heap with reserved capacity for map + heap.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            scores: HashMap::with_capacity(capacity),
            heap: BinaryHeap::with_capacity(capacity),
            seq: 0,
        }
    }

    /// Reserves capacity for at least `additional` more entries.
    pub fn reserve(&mut self, additional: usize) {
        self.scores.reserve(additional);
        self.heap.reserve(additional);
    }

    /// Shrinks internal storage to fit current contents.
    pub fn shrink_to_fit(&mut self) {
        self.scores.shrink_to_fit();
        self.heap.shrink_to_fit();
    }

    /// Clears all entries and shrinks internal storage.
    pub fn clear_shrink(&mut self) {
        self.scores.clear();
        self.heap.clear();
        self.scores.shrink_to_fit();
        self.heap.shrink_to_fit();
    }

    /// Returns the number of live keys.
    pub fn len(&self) -> usize {
        self.scores.len()
    }

    /// Returns `true` if there are no live keys.
    pub fn is_empty(&self) -> bool {
        self.scores.is_empty()
    }

    /// Returns the underlying heap length (may exceed `len()` due to stale entries).
    pub fn heap_len(&self) -> usize {
        self.heap.len()
    }

    /// Returns the current score for `key`, if present.
    pub fn score_of(&self, key: &K) -> Option<&S> {
        self.scores.get(key)
    }

    /// Updates `key`'s score and returns the previous score, if any.
    ///
    /// Pushes a new heap entry; old entries become stale and are ignored by `pop_best`.
    pub fn update(&mut self, key: K, score: S) -> Option<S> {
        let previous = self.scores.insert(key.clone(), score.clone());
        self.push_entry(key, score);
        previous
    }

    /// Removes `key` and returns its score, if present.
    ///
    /// This does not remove any stale heap entries; they will be skipped by `pop_best`.
    pub fn remove(&mut self, key: &K) -> Option<S> {
        self.scores.remove(key)
    }

    /// Pops and returns the current minimum `(key, score)`, skipping stale entries.
    pub fn pop_best(&mut self) -> Option<(K, S)> {
        loop {
            let Reverse(entry) = self.heap.pop()?;
            match self.scores.get(&entry.key) {
                Some(score) if *score == entry.score => {
                    self.scores.remove(&entry.key);
                    return Some((entry.key, entry.score));
                },
                _ => continue,
            }
        }
    }

    /// Rebuilds the heap from the authoritative `scores` map.
    pub fn rebuild(&mut self) {
        self.heap.clear();
        let entries: Vec<(K, S)> = self
            .scores
            .iter()
            .map(|(key, score)| (key.clone(), score.clone()))
            .collect();
        for (key, score) in entries {
            self.push_entry(key, score);
        }
    }

    /// Rebuilds if the heap has grown too stale relative to the map size.
    pub fn maybe_rebuild(&mut self, factor: usize) {
        let factor = factor.max(1);
        if self.heap.len() > self.scores.len().saturating_mul(factor) {
            self.rebuild();
        }
    }

    #[cfg(any(test, debug_assertions))]
    /// Returns a debug snapshot of heap/map sizes.
    pub fn debug_snapshot(&self) -> LazyHeapSnapshot {
        LazyHeapSnapshot {
            len: self.len(),
            heap_len: self.heap_len(),
        }
    }

    /// Returns an approximate memory footprint in bytes.
    pub fn approx_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.scores.capacity() * std::mem::size_of::<(K, S)>()
            + self.heap.capacity() * std::mem::size_of::<std::cmp::Reverse<HeapEntry<K, S>>>()
    }

    #[cfg(any(test, debug_assertions))]
    /// Returns a cloned view of scores for debugging.
    pub fn debug_snapshot_scores(&self) -> Vec<(K, S)>
    where
        K: Clone,
        S: Clone,
    {
        self.scores
            .iter()
            .map(|(key, score)| (key.clone(), score.clone()))
            .collect()
    }

    #[cfg(any(test, debug_assertions))]
    pub fn debug_validate_invariants(&self) {
        assert_eq!(self.len(), self.scores.len());
        if self.is_empty() {
            assert!(self.scores.is_empty());
        }
    }

    fn push_entry(&mut self, key: K, score: S) {
        let entry = HeapEntry {
            score,
            seq: self.seq,
            key,
        };
        self.seq = self.seq.wrapping_add(1);
        self.heap.push(Reverse(entry));
    }
}

#[cfg(any(test, debug_assertions))]
#[derive(Debug, PartialEq, Eq)]
pub struct LazyHeapSnapshot {
    pub len: usize,
    pub heap_len: usize,
}

impl<K, S> Default for LazyMinHeap<K, S>
where
    K: Eq + Hash + Clone,
    S: Ord + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lazy_heap_skips_stale_entries() {
        let mut heap = LazyMinHeap::new();
        heap.update("a", 5);
        heap.update("a", 2);
        heap.update("b", 3);

        assert_eq!(heap.pop_best(), Some(("a", 2)));
        assert_eq!(heap.pop_best(), Some(("b", 3)));
    }

    #[test]
    fn lazy_heap_remove_and_rebuild() {
        let mut heap = LazyMinHeap::new();
        heap.update("a", 5);
        heap.update("b", 1);
        heap.remove(&"b");
        heap.maybe_rebuild(1);
        assert_eq!(heap.pop_best(), Some(("a", 5)));
        assert_eq!(heap.pop_best(), None);
    }

    #[test]
    fn lazy_heap_update_overwrites_score_and_len() {
        let mut heap = LazyMinHeap::new();
        assert_eq!(heap.len(), 0);
        assert_eq!(heap.update("a", 10), None);
        assert_eq!(heap.len(), 1);
        assert_eq!(heap.score_of(&"a"), Some(&10));
        assert_eq!(heap.update("a", 3), Some(10));
        assert_eq!(heap.len(), 1);
        assert_eq!(heap.score_of(&"a"), Some(&3));
    }

    #[test]
    fn lazy_heap_pop_best_removes_key() {
        let mut heap = LazyMinHeap::new();
        heap.update("a", 2);
        heap.update("b", 1);
        assert_eq!(heap.pop_best(), Some(("b", 1)));
        assert_eq!(heap.score_of(&"b"), None);
        assert_eq!(heap.len(), 1);
        assert_eq!(heap.pop_best(), Some(("a", 2)));
        assert!(heap.is_empty());
    }

    #[test]
    fn lazy_heap_tie_breaks_by_seq() {
        let mut heap = LazyMinHeap::new();
        heap.update("a", 1);
        heap.update("b", 1);
        heap.update("c", 1);
        assert_eq!(heap.pop_best(), Some(("a", 1)));
        assert_eq!(heap.pop_best(), Some(("b", 1)));
        assert_eq!(heap.pop_best(), Some(("c", 1)));
    }

    #[test]
    fn lazy_heap_remove_does_not_touch_heap_until_pop() {
        let mut heap = LazyMinHeap::new();
        heap.update("a", 2);
        heap.update("b", 1);
        assert_eq!(heap.remove(&"b"), Some(1));
        assert_eq!(heap.len(), 1);
        assert_eq!(heap.pop_best(), Some(("a", 2)));
        assert_eq!(heap.pop_best(), None);
    }

    #[test]
    fn lazy_heap_rebuild_cleans_stale_entries() {
        let mut heap = LazyMinHeap::new();
        heap.update("a", 5);
        heap.update("a", 4);
        heap.update("a", 3);
        heap.update("b", 2);
        assert!(heap.heap_len() > heap.len());

        heap.rebuild();
        assert_eq!(heap.heap_len(), heap.len());
        assert_eq!(heap.pop_best(), Some(("b", 2)));
        assert_eq!(heap.pop_best(), Some(("a", 3)));
    }

    #[test]
    fn lazy_heap_maybe_rebuild_triggers_on_factor() {
        let mut heap = LazyMinHeap::new();
        heap.update("a", 3);
        heap.update("a", 2);
        heap.update("a", 1);
        heap.update("b", 4);
        assert!(heap.heap_len() > heap.len());

        heap.maybe_rebuild(1);
        assert_eq!(heap.heap_len(), heap.len());
        assert_eq!(heap.pop_best(), Some(("a", 1)));
    }

    #[test]
    fn lazy_heap_debug_invariants_hold() {
        let mut heap = LazyMinHeap::new();
        heap.update("a", 2);
        heap.update("b", 1);
        heap.remove(&"b");
        heap.debug_validate_invariants();
    }

    #[test]
    fn lazy_heap_debug_snapshots() {
        let mut heap = LazyMinHeap::new();
        heap.update("a", 2);
        heap.update("b", 1);
        let snapshot = heap.debug_snapshot();
        assert_eq!(snapshot.len, 2);
        assert!(snapshot.heap_len >= snapshot.len);

        let scores = heap.debug_snapshot_scores();
        assert_eq!(scores.len(), 2);
    }
}
