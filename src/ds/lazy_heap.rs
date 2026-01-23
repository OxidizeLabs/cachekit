//! Lazy min-heap with stale entry skipping.
//!
//! A priority queue that supports O(1) updates by deferring cleanup. Instead
//! of modifying heap entries in place, updates push new entries and mark old
//! ones as stale. The [`pop_best`](LazyMinHeap::pop_best) operation skips
//! stale entries automatically.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                         LazyMinHeap Layout                                  │
//! │                                                                             │
//! │   ┌───────────────────────────────────────────────────────────────────┐    │
//! │   │  scores: HashMap<K, S>   (authoritative source of truth)          │    │
//! │   │                                                                   │    │
//! │   │    ┌─────────┬─────────┐                                         │    │
//! │   │    │  key    │  score  │                                         │    │
//! │   │    ├─────────┼─────────┤                                         │    │
//! │   │    │  "A"    │   10    │                                         │    │
//! │   │    │  "B"    │    3    │                                         │    │
//! │   │    │  "C"    │    7    │                                         │    │
//! │   │    └─────────┴─────────┘                                         │    │
//! │   │                                                                   │    │
//! │   │    len() = 3 (live entries)                                      │    │
//! │   └───────────────────────────────────────────────────────────────────┘    │
//! │                                                                             │
//! │   ┌───────────────────────────────────────────────────────────────────┐    │
//! │   │  heap: BinaryHeap<Reverse<HeapEntry>>   (may have stale entries) │    │
//! │   │                                                                   │    │
//! │   │    Min-heap order (smallest score first):                        │    │
//! │   │                                                                   │    │
//! │   │    ┌────────────────────────────────────────────────────────┐   │    │
//! │   │    │ ("B", 3, seq=5)  ← current min, matches scores["B"]    │   │    │
//! │   │    │ ("C", 7, seq=4)  ← valid                                │   │    │
//! │   │    │ ("A", 10, seq=3) ← valid                                │   │    │
//! │   │    │ ("A", 15, seq=1) ← STALE: scores["A"]=10, not 15       │   │    │
//! │   │    │ ("B", 8, seq=2)  ← STALE: scores["B"]=3, not 8         │   │    │
//! │   │    └────────────────────────────────────────────────────────┘   │    │
//! │   │                                                                   │    │
//! │   │    heap_len() = 5 (includes stale entries)                       │    │
//! │   └───────────────────────────────────────────────────────────────────┘    │
//! │                                                                             │
//! │   seq: 6  (monotonic counter for tie-breaking)                             │
//! └─────────────────────────────────────────────────────────────────────────────┘
//!
//! Update Flow
//! ───────────
//!   update("A", 10):
//!     1. scores["A"] = 10          (authoritative update)
//!     2. heap.push(("A", 10, seq)) (new entry, old entries become stale)
//!     3. seq += 1
//!
//! Pop Flow
//! ────────
//!   pop_best():
//!     loop:
//!       entry = heap.pop()         → ("A", 15, seq=1)
//!       if scores["A"] == 15?      → No! scores["A"]=10
//!         skip (stale)
//!       ...
//!       entry = heap.pop()         → ("B", 3, seq=5)
//!       if scores["B"] == 3?       → Yes!
//!         scores.remove("B")
//!         return ("B", 3)
//!
//! Rebuild
//! ───────
//!   When heap_len >> len(), call rebuild() to clear stale entries:
//!     heap.clear()
//!     for (key, score) in scores:
//!       heap.push((key, score, seq++))
//! ```
//!
//! ## Key Concepts
//!
//! - **Lazy deletion**: Old heap entries aren't removed; they're skipped when
//!   their score doesn't match the authoritative `scores` map
//! - **Sequence numbers**: Break ties for equal scores (FIFO order)
//! - **Periodic rebuild**: When stale entries accumulate, `rebuild()` or
//!   `maybe_rebuild()` cleans up the heap
//!
//! ## Operations
//!
//! | Operation      | Description                           | Complexity      |
//! |----------------|---------------------------------------|-----------------|
//! | `update`       | Set/update score, push heap entry     | O(log n)        |
//! | `remove`       | Remove from scores map only           | O(1)            |
//! | `pop_best`     | Pop min, skipping stale entries       | Amortized O(log n) |
//! | `score_of`     | Get current score for key             | O(1)            |
//! | `rebuild`      | Rebuild heap from scores map          | O(n log n)      |
//! | `maybe_rebuild`| Rebuild if heap too stale             | O(1) or O(n log n) |
//!
//! ## Use Cases
//!
//! - **LFU eviction**: Track access frequencies, pop least-frequently-used
//! - **Priority scheduling**: Tasks with changing priorities
//! - **Expiration tracking**: Items with updatable TTLs
//!
//! ## Example Usage
//!
//! ```
//! use cachekit::ds::LazyMinHeap;
//!
//! let mut heap: LazyMinHeap<&str, u32> = LazyMinHeap::new();
//!
//! // Insert items with scores (lower = higher priority)
//! heap.update("task_a", 5);
//! heap.update("task_b", 2);
//! heap.update("task_c", 8);
//!
//! // Update a score (creates stale entry, doesn't remove old one)
//! heap.update("task_a", 1);  // task_a now has priority 1
//!
//! // Pop returns minimum score, skipping stale entries
//! assert_eq!(heap.pop_best(), Some(("task_a", 1)));
//! assert_eq!(heap.pop_best(), Some(("task_b", 2)));
//! assert_eq!(heap.pop_best(), Some(("task_c", 8)));
//! assert_eq!(heap.pop_best(), None);
//! ```
//!
//! ## Performance Trade-offs
//!
//! - **Fast updates**: O(log n) push, no removal needed
//! - **Memory overhead**: Stale entries consume space until rebuilt
//! - **Rebuild cost**: O(n log n) but only when heap grows too stale
//!
//! ## Thread Safety
//!
//! `LazyMinHeap` is not thread-safe. Wrap in a mutex for concurrent access.
//!
//! ## Implementation Notes
//!
//! - Uses `BinaryHeap<Reverse<_>>` for min-heap behavior
//! - Tie-breaking uses sequence numbers for FIFO among equal scores
//! - `debug_validate_invariants()` available in debug/test builds
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};
use std::hash::Hash;

#[derive(Debug, Clone)]
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

/// Min-heap with O(1) score updates via lazy deletion.
///
/// Maintains an authoritative `scores` map and a heap that may contain stale
/// entries. Updates modify the map and push new heap entries; old entries
/// are skipped during [`pop_best`](Self::pop_best).
///
/// # Type Parameters
///
/// - `K`: Key type (must be `Eq + Hash + Clone`)
/// - `S`: Score type (must be `Ord + Clone`)
///
/// # Example
///
/// ```
/// use cachekit::ds::LazyMinHeap;
///
/// let mut heap: LazyMinHeap<&str, i32> = LazyMinHeap::new();
///
/// // Track item priorities
/// heap.update("low", 10);
/// heap.update("high", 1);
/// heap.update("medium", 5);
///
/// // Pop in priority order (lowest score first)
/// assert_eq!(heap.pop_best(), Some(("high", 1)));
/// assert_eq!(heap.pop_best(), Some(("medium", 5)));
/// assert_eq!(heap.pop_best(), Some(("low", 10)));
/// ```
///
/// # Use Case: LFU Cache Eviction
///
/// ```
/// use cachekit::ds::LazyMinHeap;
///
/// // Track access counts (lower = less frequently used)
/// let mut freq: LazyMinHeap<&str, u32> = LazyMinHeap::new();
///
/// // Record accesses
/// freq.update("page_a", 1);
/// freq.update("page_b", 1);
/// freq.update("page_a", 2);  // accessed again
/// freq.update("page_c", 1);
/// freq.update("page_a", 3);  // accessed again
///
/// // Evict least frequently used
/// let (victim, _count) = freq.pop_best().unwrap();
/// assert!(victim == "page_b" || victim == "page_c");  // Both have count 1
/// ```
#[derive(Debug)]
pub struct LazyMinHeap<K, S> {
    scores: HashMap<K, ScoreEntry<S>>,
    heap: BinaryHeap<Reverse<HeapEntry<K, S>>>,
    seq: u64,
}

impl<K, S> LazyMinHeap<K, S>
where
    K: Eq + Hash + Clone,
    S: Ord + Clone,
{
    /// Creates an empty heap.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::LazyMinHeap;
    ///
    /// let heap: LazyMinHeap<String, u32> = LazyMinHeap::new();
    /// assert!(heap.is_empty());
    /// ```
    pub fn new() -> Self {
        Self {
            scores: HashMap::new(),
            heap: BinaryHeap::new(),
            seq: 0,
        }
    }

    /// Creates an empty heap with pre-allocated capacity.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::LazyMinHeap;
    ///
    /// let heap: LazyMinHeap<i32, i32> = LazyMinHeap::with_capacity(1000);
    /// assert!(heap.is_empty());
    /// ```
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            scores: HashMap::with_capacity(capacity),
            heap: BinaryHeap::with_capacity(capacity),
            seq: 0,
        }
    }

    /// Reserves capacity for at least `additional` more entries.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::LazyMinHeap;
    ///
    /// let mut heap: LazyMinHeap<i32, i32> = LazyMinHeap::new();
    /// heap.reserve(100);
    /// ```
    pub fn reserve(&mut self, additional: usize) {
        self.scores.reserve(additional);
        self.heap.reserve(additional);
    }

    /// Shrinks internal storage to fit current contents.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::LazyMinHeap;
    ///
    /// let mut heap: LazyMinHeap<i32, i32> = LazyMinHeap::with_capacity(1000);
    /// heap.update(1, 10);
    /// heap.shrink_to_fit();
    /// ```
    pub fn shrink_to_fit(&mut self) {
        self.scores.shrink_to_fit();
        self.heap.shrink_to_fit();
    }

    /// Clears all entries and shrinks internal storage.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::LazyMinHeap;
    ///
    /// let mut heap: LazyMinHeap<&str, i32> = LazyMinHeap::new();
    /// heap.update("a", 1);
    /// heap.update("b", 2);
    ///
    /// heap.clear_shrink();
    /// assert!(heap.is_empty());
    /// ```
    pub fn clear_shrink(&mut self) {
        self.scores.clear();
        self.heap.clear();
        self.scores.shrink_to_fit();
        self.heap.shrink_to_fit();
    }

    /// Returns the number of live keys.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::LazyMinHeap;
    ///
    /// let mut heap: LazyMinHeap<&str, i32> = LazyMinHeap::new();
    /// assert_eq!(heap.len(), 0);
    ///
    /// heap.update("a", 1);
    /// heap.update("b", 2);
    /// assert_eq!(heap.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        self.scores.len()
    }

    /// Returns `true` if there are no live keys.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::LazyMinHeap;
    ///
    /// let mut heap: LazyMinHeap<i32, i32> = LazyMinHeap::new();
    /// assert!(heap.is_empty());
    ///
    /// heap.update(1, 10);
    /// assert!(!heap.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.scores.is_empty()
    }

    /// Returns the underlying heap length (may exceed `len()` due to stale entries).
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::LazyMinHeap;
    ///
    /// let mut heap: LazyMinHeap<&str, i32> = LazyMinHeap::new();
    /// heap.update("a", 5);
    /// heap.update("a", 3);  // Creates stale entry
    /// heap.update("a", 1);  // Creates another stale entry
    ///
    /// assert_eq!(heap.len(), 1);       // 1 live key
    /// assert_eq!(heap.heap_len(), 3);  // 3 heap entries (2 stale)
    /// ```
    pub fn heap_len(&self) -> usize {
        self.heap.len()
    }

    /// Returns the current score for `key`, if present.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::LazyMinHeap;
    ///
    /// let mut heap: LazyMinHeap<&str, i32> = LazyMinHeap::new();
    /// heap.update("task", 5);
    ///
    /// assert_eq!(heap.score_of(&"task"), Some(&5));
    /// assert_eq!(heap.score_of(&"missing"), None);
    /// ```
    pub fn score_of(&self, key: &K) -> Option<&S> {
        self.scores.get(key).map(|entry| &entry.score)
    }

    /// Updates `key`'s score and returns the previous score, if any.
    ///
    /// Pushes a new heap entry; old entries become stale and are skipped
    /// by [`pop_best`](Self::pop_best).
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::LazyMinHeap;
    ///
    /// let mut heap: LazyMinHeap<&str, i32> = LazyMinHeap::new();
    ///
    /// // First insert
    /// assert_eq!(heap.update("item", 10), None);
    ///
    /// // Update returns old score
    /// assert_eq!(heap.update("item", 5), Some(10));
    /// assert_eq!(heap.score_of(&"item"), Some(&5));
    /// ```
    pub fn update(&mut self, key: K, score: S) -> Option<S> {
        let seq = self.seq;
        self.seq = self.seq.wrapping_add(1);
        let previous = self.scores.insert(
            key.clone(),
            ScoreEntry {
                score: score.clone(),
                seq,
            },
        );
        self.push_entry_with_seq(key, score, seq);
        previous.map(|entry| entry.score)
    }

    /// Removes `key` and returns its score, if present.
    ///
    /// This only removes from the authoritative map; stale heap entries
    /// will be skipped by [`pop_best`](Self::pop_best).
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::LazyMinHeap;
    ///
    /// let mut heap: LazyMinHeap<&str, i32> = LazyMinHeap::new();
    /// heap.update("a", 1);
    /// heap.update("b", 2);
    ///
    /// assert_eq!(heap.remove(&"a"), Some(1));
    /// assert_eq!(heap.remove(&"a"), None);  // Already removed
    ///
    /// // "b" is still there
    /// assert_eq!(heap.pop_best(), Some(("b", 2)));
    /// ```
    pub fn remove(&mut self, key: &K) -> Option<S> {
        self.scores.remove(key).map(|entry| entry.score)
    }

    /// Pops and returns the minimum `(key, score)`, skipping stale entries.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::LazyMinHeap;
    ///
    /// let mut heap: LazyMinHeap<&str, i32> = LazyMinHeap::new();
    /// heap.update("high", 1);
    /// heap.update("low", 10);
    ///
    /// // Returns minimum score first
    /// assert_eq!(heap.pop_best(), Some(("high", 1)));
    /// assert_eq!(heap.pop_best(), Some(("low", 10)));
    /// assert_eq!(heap.pop_best(), None);
    /// ```
    pub fn pop_best(&mut self) -> Option<(K, S)> {
        loop {
            let Reverse(entry) = self.heap.pop()?;
            match self.scores.get(&entry.key) {
                Some(current) if current.score == entry.score && current.seq == entry.seq => {
                    self.scores.remove(&entry.key);
                    return Some((entry.key, entry.score));
                },
                _ => continue,
            }
        }
    }

    /// Rebuilds the heap from the authoritative `scores` map.
    ///
    /// Removes all stale entries. Call this periodically or when
    /// `heap_len()` greatly exceeds `len()`.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::LazyMinHeap;
    ///
    /// let mut heap: LazyMinHeap<&str, i32> = LazyMinHeap::new();
    ///
    /// // Create many stale entries
    /// for i in 0..10 {
    ///     heap.update("key", i);
    /// }
    /// assert_eq!(heap.len(), 1);
    /// assert_eq!(heap.heap_len(), 10);  // 9 stale entries
    ///
    /// heap.rebuild();
    /// assert_eq!(heap.heap_len(), 1);   // Stale entries removed
    /// ```
    pub fn rebuild(&mut self) {
        self.heap.clear();
        let entries: Vec<(K, ScoreEntry<S>)> = self
            .scores
            .iter()
            .map(|(key, entry)| (key.clone(), entry.clone()))
            .collect();
        for (key, entry) in entries {
            self.push_entry_with_seq(key, entry.score, entry.seq);
        }
    }

    /// Rebuilds if the heap has grown too stale relative to map size.
    ///
    /// Triggers rebuild when `heap_len() > len() * factor`.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::LazyMinHeap;
    ///
    /// let mut heap: LazyMinHeap<&str, i32> = LazyMinHeap::new();
    /// heap.update("a", 1);
    /// heap.update("a", 2);
    /// heap.update("a", 3);  // heap_len=3, len=1
    ///
    /// // Rebuild if heap_len > len * 2
    /// heap.maybe_rebuild(2);
    /// assert_eq!(heap.heap_len(), 1);
    /// ```
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
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::LazyMinHeap;
    ///
    /// let heap: LazyMinHeap<u64, u64> = LazyMinHeap::with_capacity(100);
    /// let bytes = heap.approx_bytes();
    /// assert!(bytes > 0);
    /// ```
    pub fn approx_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.scores.capacity() * std::mem::size_of::<(K, ScoreEntry<S>)>()
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
            .map(|(key, entry)| (key.clone(), entry.score.clone()))
            .collect()
    }

    #[cfg(any(test, debug_assertions))]
    /// Validates internal invariants (debug/test builds only).
    pub fn debug_validate_invariants(&self) {
        assert_eq!(self.len(), self.scores.len());
        if self.is_empty() {
            assert!(self.scores.is_empty());
        }
    }

    fn push_entry_with_seq(&mut self, key: K, score: S, seq: u64) {
        let entry = HeapEntry { score, seq, key };
        self.heap.push(Reverse(entry));
    }
}

#[derive(Debug, Clone)]
struct ScoreEntry<S> {
    score: S,
    seq: u64,
}

#[cfg(any(test, debug_assertions))]
#[derive(Debug, Clone, PartialEq, Eq)]
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
    fn lazy_heap_update_same_score_refreshes_order() {
        let mut heap = LazyMinHeap::new();
        heap.update("a", 1);
        heap.update("b", 1);
        heap.update("a", 1); // refresh "a" to the back of the equal-score queue
        assert_eq!(heap.pop_best(), Some(("b", 1)));
        assert_eq!(heap.pop_best(), Some(("a", 1)));
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

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    // =============================================================================
    // Property Tests - Min-Heap Ordering
    // =============================================================================

    proptest! {
        /// Property: pop_best returns items in ascending score order
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_min_heap_ordering(
            entries in prop::collection::vec((any::<u32>(), any::<u32>()), 0..50)
        ) {
            let mut heap = LazyMinHeap::new();

            // Insert entries
            for (key, score) in entries {
                heap.update(key, score);
            }

            // Pop all - scores should be in ascending order
            let mut last_score = None;
            while let Some((_key, score)) = heap.pop_best() {
                if let Some(prev_score) = last_score {
                    prop_assert!(score >= prev_score);
                }
                last_score = Some(score);
            }

            prop_assert!(heap.is_empty());
        }

        /// Property: pop_best with tie-breaking uses FIFO order
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_tie_breaking_fifo(
            keys in prop::collection::vec(any::<u32>(), 3..20)
        ) {
            let mut heap = LazyMinHeap::new();
            let score = 1u32; // Same score for all

            // Insert all with same score
            for key in &keys {
                heap.update(*key, score);
            }

            // Pop should return in insertion order (FIFO)
            for expected_key in keys {
                if let Some((key, s)) = heap.pop_best() {
                    prop_assert_eq!(s, score);
                    prop_assert_eq!(key, expected_key);
                }
            }
        }
    }

    // =============================================================================
    // Property Tests - Update Operations
    // =============================================================================

    proptest! {
        /// Property: update overwrites previous score
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_update_overwrites(
            key in any::<u32>(),
            scores in prop::collection::vec(any::<u32>(), 1..20)
        ) {
            let mut heap = LazyMinHeap::new();

            // Update same key multiple times
            for score in &scores {
                heap.update(key, *score);
                prop_assert_eq!(heap.score_of(&key), Some(score));
                prop_assert_eq!(heap.len(), 1);
            }

            // Pop should return the last score
            let popped = heap.pop_best();
            prop_assert!(popped.is_some());
            let (k, s) = popped.unwrap();
            prop_assert_eq!(k, key);
            prop_assert_eq!(s, *scores.last().unwrap());
        }

        /// Property: update with same score is idempotent
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_update_idempotent(
            key in any::<u32>(),
            score in any::<u32>(),
            repeat_count in 1usize..10
        ) {
            let mut heap = LazyMinHeap::new();

            for _ in 0..repeat_count {
                heap.update(key, score);
                prop_assert_eq!(heap.score_of(&key), Some(&score));
                prop_assert_eq!(heap.len(), 1);
            }
        }

        /// Property: update returns old score
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_update_returns_old_score(
            key in any::<u32>(),
            score1 in any::<u32>(),
            score2 in any::<u32>()
        ) {
            let mut heap = LazyMinHeap::new();

            let old = heap.update(key, score1);
            prop_assert_eq!(old, None);

            let old = heap.update(key, score2);
            prop_assert_eq!(old, Some(score1));
        }
    }

    // =============================================================================
    // Property Tests - Remove Operations
    // =============================================================================

    proptest! {
        /// Property: remove decreases length by 1
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_remove_decreases_length(
            entries in prop::collection::vec((any::<u32>(), any::<u32>()), 1..30)
        ) {
            let mut heap = LazyMinHeap::new();

            // Insert entries
            for (key, score) in &entries {
                heap.update(*key, *score);
            }

            // Remove each key
            for (key, score) in entries {
                let old_len = heap.len();
                let removed = heap.remove(&key);

                if removed == Some(score) {
                    prop_assert_eq!(heap.len(), old_len - 1);
                    prop_assert_eq!(heap.score_of(&key), None);
                }
            }
        }

        /// Property: remove makes key unavailable
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_remove_makes_unavailable(
            key in any::<u32>(),
            score in any::<u32>()
        ) {
            let mut heap = LazyMinHeap::new();
            heap.update(key, score);

            prop_assert_eq!(heap.score_of(&key), Some(&score));

            let removed = heap.remove(&key);
            prop_assert_eq!(removed, Some(score));
            prop_assert_eq!(heap.score_of(&key), None);
            prop_assert!(heap.is_empty());
        }

        /// Property: removing non-existent key returns None
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_remove_missing_returns_none(
            insert_keys in prop::collection::vec(0u32..20, 1..10),
            query_key in 20u32..40
        ) {
            let mut heap = LazyMinHeap::new();

            for key in insert_keys {
                heap.update(key, 1);
            }

            let removed = heap.remove(&query_key);
            prop_assert_eq!(removed, None);
        }
    }

    // =============================================================================
    // Property Tests - Pop Operations
    // =============================================================================

    proptest! {
        /// Property: pop_best decreases length by 1
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_pop_decreases_length(
            entries in prop::collection::vec((any::<u32>(), any::<u32>()), 1..30)
        ) {
            let mut heap = LazyMinHeap::new();

            for (key, score) in entries {
                heap.update(key, score);
            }

            while !heap.is_empty() {
                let old_len = heap.len();
                let popped = heap.pop_best();

                prop_assert!(popped.is_some());
                prop_assert_eq!(heap.len(), old_len - 1);
            }

            prop_assert_eq!(heap.pop_best(), None);
        }

        /// Property: pop_best removes key from scores
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_pop_removes_key(
            entries in prop::collection::vec((any::<u32>(), any::<u32>()), 1..30)
        ) {
            let mut heap = LazyMinHeap::new();

            for (key, score) in entries {
                heap.update(key, score);
            }

            while let Some((key, _score)) = heap.pop_best() {
                prop_assert_eq!(heap.score_of(&key), None);
            }
        }

        /// Property: pop_best on empty returns None
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_pop_empty_returns_none(_unit in any::<()>()) {
            let mut heap: LazyMinHeap<u32, u32> = LazyMinHeap::new();
            prop_assert_eq!(heap.pop_best(), None);
        }
    }

    // =============================================================================
    // Property Tests - Stale Entry Handling
    // =============================================================================

    proptest! {
        /// Property: stale entries are skipped during pop_best
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_stale_entries_skipped(
            updates in prop::collection::vec((0u32..10, any::<u32>()), 10..50)
        ) {
            let mut heap = LazyMinHeap::new();

            // Insert many updates to create stale entries
            for (key, score) in updates {
                heap.update(key, score);
            }

            // Each key should only be popped once
            let mut seen_keys = std::collections::HashSet::new();

            while let Some((key, _score)) = heap.pop_best() {
                prop_assert!(!seen_keys.contains(&key));
                seen_keys.insert(key);
            }

            prop_assert!(heap.is_empty());
        }
    }

    // =============================================================================
    // Property Tests - Rebuild Operations
    // =============================================================================

    proptest! {
        /// Property: rebuild preserves length and order
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_rebuild_preserves_order(
            updates in prop::collection::vec((0u32..20, any::<u32>()), 10..50)
        ) {
            let mut heap = LazyMinHeap::new();

            // Insert with updates to create stale entries
            for (key, score) in updates {
                heap.update(key, score);
            }

            let len_before = heap.len();

            // Rebuild
            heap.rebuild();

            // Length should be preserved
            prop_assert_eq!(heap.len(), len_before);

            // heap_len should now equal len (no stale entries)
            prop_assert_eq!(heap.heap_len(), heap.len());

            // Pop order should still be ascending
            let mut last_score = None;
            while let Some((_key, score)) = heap.pop_best() {
                if let Some(prev_score) = last_score {
                    prop_assert!(score >= prev_score);
                }
                last_score = Some(score);
            }
        }

        /// Property: maybe_rebuild with factor triggers correctly
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_maybe_rebuild_factor(
            key in any::<u32>(),
            updates in prop::collection::vec(any::<u32>(), 3..20)
        ) {
            let mut heap = LazyMinHeap::new();

            // Update same key multiple times to create stale entries
            for score in updates {
                heap.update(key, score);
            }

            let heap_len_before = heap.heap_len();
            let len = heap.len();

            // maybe_rebuild with factor 1 should always rebuild if heap_len > len
            if heap_len_before > len {
                heap.maybe_rebuild(1);
                prop_assert_eq!(heap.heap_len(), heap.len());
            }
        }
    }

    // =============================================================================
    // Property Tests - Length and Empty State
    // =============================================================================

    proptest! {
        /// Property: len tracks number of unique keys
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_len_tracks_unique_keys(
            entries in prop::collection::vec((any::<u32>(), any::<u32>()), 0..50)
        ) {
            let mut heap = LazyMinHeap::new();

            for (key, score) in &entries {
                heap.update(*key, *score);
            }

            let unique_count = {
                let mut unique = std::collections::HashSet::new();
                for (key, _) in entries {
                    unique.insert(key);
                }
                unique.len()
            };

            prop_assert_eq!(heap.len(), unique_count);
        }

        /// Property: is_empty is consistent with len
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_is_empty_consistent(
            entries in prop::collection::vec((any::<u32>(), any::<u32>()), 0..30)
        ) {
            let mut heap = LazyMinHeap::new();

            for (key, score) in entries {
                heap.update(key, score);

                if heap.is_empty() {
                    prop_assert_eq!(heap.len(), 0);
                } else {
                    prop_assert!(!heap.is_empty());
                }
            }
        }
    }

    // =============================================================================
    // Property Tests - Score Queries
    // =============================================================================

    proptest! {
        /// Property: score_of returns current score
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_score_of_returns_current(
            entries in prop::collection::vec((any::<u32>(), any::<u32>()), 1..30)
        ) {
            let mut heap = LazyMinHeap::new();

            for (key, score) in &entries {
                heap.update(*key, *score);
            }

            // Verify score_of for all keys
            for (key, expected_score) in entries {
                if let Some(&actual_score) = heap.score_of(&key) {
                    prop_assert_eq!(actual_score, expected_score);
                }
            }
        }

        /// Property: score_of returns None for removed keys
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_score_of_removed_is_none(
            key in any::<u32>(),
            score in any::<u32>()
        ) {
            let mut heap = LazyMinHeap::new();
            heap.update(key, score);
            heap.remove(&key);

            prop_assert_eq!(heap.score_of(&key), None);
        }
    }

    // =============================================================================
    // Property Tests - Clear Operations
    // =============================================================================

    proptest! {
        /// Property: clear_shrink resets to empty state
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_clear_resets_state(
            entries in prop::collection::vec((any::<u32>(), any::<u32>()), 1..30)
        ) {
            let mut heap = LazyMinHeap::new();

            for (key, score) in entries {
                heap.update(key, score);
            }

            heap.clear_shrink();

            prop_assert!(heap.is_empty());
            prop_assert_eq!(heap.len(), 0);
            prop_assert_eq!(heap.pop_best(), None);
        }

        /// Property: usable after clear
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_usable_after_clear(
            entries1 in prop::collection::vec((any::<u32>(), any::<u32>()), 1..20),
            entries2 in prop::collection::vec((any::<u32>(), any::<u32>()), 1..20)
        ) {
            let mut heap = LazyMinHeap::new();

            for (key, score) in entries1 {
                heap.update(key, score);
            }

            heap.clear_shrink();

            // Should be usable after clear
            for (key, score) in &entries2 {
                heap.update(*key, *score);
            }

            let unique_count = {
                let mut unique = std::collections::HashSet::new();
                for (key, _) in entries2 {
                    unique.insert(key);
                }
                unique.len()
            };

            prop_assert_eq!(heap.len(), unique_count);
        }
    }

    // =============================================================================
    // Property Tests - Reference Implementation Equivalence
    // =============================================================================

    proptest! {
        /// Property: Behavior matches reference BinaryHeap for basic operations
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_matches_binary_heap(
            operations in prop::collection::vec((0u8..3, any::<u32>(), any::<u32>()), 0..50)
        ) {
            let mut heap = LazyMinHeap::new();
            let mut reference = std::collections::BinaryHeap::new();
            let mut live_keys = std::collections::HashSet::new();
            use std::cmp::Reverse;

            for (op, key, score) in operations {
                match op % 3 {
                    0 => {
                        // update
                        heap.update(key, score);

                        // Update reference: remove old, add new
                        reference.retain(|&Reverse((_s, k))| k != key);
                        reference.push(Reverse((score, key)));
                        live_keys.insert(key);
                    }
                    1 => {
                        // pop_best
                        let heap_val = heap.pop_best();

                        // Find min in reference that's still live
                        let mut ref_val = None;
                        while let Some(Reverse((score, key))) = reference.pop() {
                            if live_keys.contains(&key) {
                                ref_val = Some((key, score));
                                live_keys.remove(&key);
                                break;
                            }
                        }

                        prop_assert_eq!(heap_val, ref_val);
                    }
                    2 => {
                        // remove
                        heap.remove(&key);
                        live_keys.remove(&key);
                    }
                    _ => unreachable!(),
                }

                // Verify consistency
                prop_assert_eq!(heap.len(), live_keys.len());
                prop_assert_eq!(heap.is_empty(), live_keys.is_empty());
            }
        }
    }
}
