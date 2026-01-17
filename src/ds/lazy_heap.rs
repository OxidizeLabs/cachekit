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
        self.scores.get(key)
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
        let previous = self.scores.insert(key.clone(), score.clone());
        self.push_entry(key, score);
        previous
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
        self.scores.remove(key)
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
                Some(score) if *score == entry.score => {
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
        let entries: Vec<(K, S)> = self
            .scores
            .iter()
            .map(|(key, score)| (key.clone(), score.clone()))
            .collect();
        for (key, score) in entries {
            self.push_entry(key, score);
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
    /// Validates internal invariants (debug/test builds only).
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
