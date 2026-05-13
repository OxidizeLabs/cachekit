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
//! ┌────────────────────────────────────────────────────────────────────────────┐
//! │                         LazyMinHeap Layout                                 │
//! │                                                                            │
//! │   ┌───────────────────────────────────────────────────────────────────┐    │
//! │   │  scores: HashMap<K, S>   (authoritative source of truth)          │    │
//! │   │                                                                   │    │
//! │   │    ┌─────────┬─────────┐                                          │    │
//! │   │    │  key    │  score  │                                          │    │
//! │   │    ├─────────┼─────────┤                                          │    │
//! │   │    │  "A"    │   10    │                                          │    │
//! │   │    │  "B"    │    3    │                                          │    │
//! │   │    │  "C"    │    7    │                                          │    │
//! │   │    └─────────┴─────────┘                                          │    │
//! │   │                                                                   │    │
//! │   │    len() = 3 (live entries)                                       │    │
//! │   └───────────────────────────────────────────────────────────────────┘    │
//! │                                                                            │
//! │   ┌───────────────────────────────────────────────────────────────────┐    │
//! │   │  heap: BinaryHeap<Reverse<HeapEntry>>   (may have stale entries)  │    │
//! │   │                                                                   │    │
//! │   │    Min-heap order (smallest score first):                         │    │
//! │   │                                                                   │    │
//! │   │    ┌────────────────────────────────────────────────────────┐     │    │
//! │   │    │ ("B", 3, seq=5)  ← current min, matches scores["B"]    │     │    │
//! │   │    │ ("C", 7, seq=4)  ← valid                               │     │    │
//! │   │    │ ("A", 10, seq=3) ← valid                               │     │    │
//! │   │    │ ("A", 15, seq=1) ← STALE: scores["A"]=10, not 15       │     │    │
//! │   │    │ ("B", 8, seq=2)  ← STALE: scores["B"]=3, not 8         │     │    │
//! │   │    └────────────────────────────────────────────────────────┘     │    │
//! │   │                                                                   │    │
//! │   │    heap_len() = 5 (includes stale entries)                        │    │
//! │   └───────────────────────────────────────────────────────────────────┘    │
//! │                                                                            │
//! │   seq: 6  (monotonic counter for tie-breaking)                             │
//! └────────────────────────────────────────────────────────────────────────────┘
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
//! | Operation      | Description                           | Complexity         |
//! |----------------|---------------------------------------|--------------------|
//! | `update`       | Set/update score, push heap entry     | O(log n)           |
//! | `remove`       | Remove from scores map only           | O(1)               |
//! | `pop_best`     | Pop min, skipping stale entries       | Amortized O(log n) |
//! | `score_of`     | Get current score for key             | O(1)               |
//! | `rebuild`      | Rebuild heap from scores map          | O(n log n)         |
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
//! ## Security Considerations
//!
//! `LazyMinHeap` is intended for internal bookkeeping inside eviction
//! policies and similar trusted subsystems. Calls are assumed to come
//! from in-process code, **not** directly from adversary-controlled
//! input. The hardening below addresses the exposure paths that remain
//! when the key set or the call volume is influenced by untrusted
//! input.
//!
//! - **Unbounded memory growth via stale entries.** [`update`] pushes a
//!   new heap entry on every call; the old entry is not removed, it is
//!   skipped by [`pop_best`]. An attacker that can drive
//!   [`update`] on the same (small) key set faster than the caller
//!   drains via [`pop_best`] or
//!   [`rebuild`] can grow the heap without bound even though
//!   [`len`](LazyMinHeap::len) stays tiny. Pair the heap with a bounded
//!   rebuild cadence — either call [`maybe_rebuild`] on a schedule, or
//!   construct the heap with
//!   [`LazyMinHeap::with_auto_rebuild`] /
//!   [`LazyMinHeap::set_auto_rebuild`] so every [`update`] runs a
//!   [`maybe_rebuild`] with the configured factor.
//! - **Constructor DoS via oversized capacity.** [`with_capacity`],
//!   [`reserve`], and [`from_iter`] forward their argument straight
//!   into `HashMap` / `BinaryHeap` allocation. A capacity close to
//!   `usize::MAX` would abort the process inside the allocator. Both
//!   public constructors now reject capacities above
//!   [`MAX_CAPACITY`]. For attacker-influenced capacity, prefer the
//!   fallible [`try_with_capacity`] /
//!   [`try_reserve`] variants, which surface the error as a
//!   [`LazyMinHeapError`] instead of aborting.
//! - **Sequence-number wraparound.** The [`pop_best`] staleness check
//!   relies on `(score, seq)` being unique across live and stale heap
//!   entries for the same key. The counter is `u64`, so wrap is only
//!   reachable after ≈ 2⁶⁴ `update` calls, but a wrap would let a
//!   stale heap entry satisfy the equality check and be popped as if
//!   live. The counter is now guarded by `checked_add`; on imminent
//!   overflow the heap renumbers every live entry with fresh
//!   sequential seqs and resets the counter, preserving FIFO
//!   tie-breaking order.
//! - **`approx_bytes` overflow.** The old formula `capacity *
//!   size_of::<…>()` could overflow `usize` for pathologically large
//!   capacities. `approx_bytes` now uses `saturating_mul` /
//!   `saturating_add` and returns `usize::MAX` in the saturating case
//!   rather than panicking in debug or wrapping silently in release.
//! - **`Debug` output leaks keys and scores.** The derived `Debug`
//!   recursed through every `(key, score)` pair and every stale heap
//!   entry, turning `tracing::debug!`, `dbg!`, and panic-unwind
//!   backtraces into a disclosure channel for caches keyed on session
//!   tokens / API keys. The impl is now hand-written and redacts every
//!   stored key and score, reporting only `len`, `heap_len`, and
//!   whether auto-rebuild is configured. Callers that need full
//!   contents can iterate via [`iter`](LazyMinHeap::iter) or
//!   [`into_iter`](LazyMinHeap#impl-IntoIterator) and log entries they
//!   have vetted.
//!
//! Thread-safety, timing side channels from the backing `HashMap`, and
//! the lack of bytes-level budgeting mirror
//! [`ClockRing`](crate::ds::ClockRing); consult its module docs for
//! the same set of caveats.
//!
//! ## Implementation Notes
//!
//! - Uses `BinaryHeap<Reverse<_>>` for min-heap behavior
//! - Tie-breaking uses sequence numbers for FIFO among equal scores
//!
//! [`update`]: LazyMinHeap::update
//! [`pop_best`]: LazyMinHeap::pop_best
//! [`rebuild`]: LazyMinHeap::rebuild
//! [`maybe_rebuild`]: LazyMinHeap::maybe_rebuild
//! [`len`]: LazyMinHeap::len
//! [`with_capacity`]: LazyMinHeap::with_capacity
//! [`reserve`]: LazyMinHeap::reserve
//! [`try_with_capacity`]: LazyMinHeap::try_with_capacity
//! [`try_reserve`]: LazyMinHeap::try_reserve
//! [`from_iter`]: LazyMinHeap#impl-FromIterator%3C(K,+S)%3E
use std::borrow::Borrow;
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};
use std::fmt;
use std::hash::Hash;
use std::iter::FusedIterator;

/// Coarse upper bound on `capacity` accepted by
/// [`LazyMinHeap::with_capacity`] / [`LazyMinHeap::try_with_capacity`]
/// and [`LazyMinHeap::reserve`] / [`LazyMinHeap::try_reserve`].
///
/// This is a *first* guard that rejects obviously pathological values
/// cheaply. It is intentionally permissive: for any key/score size
/// larger than a few bytes, allocating `MAX_CAPACITY` entries would
/// still exhaust memory. The fallible constructors defend against
/// that separately by using `HashMap::try_reserve` / `Vec::try_reserve`,
/// so out-of-memory conditions surface as
/// [`LazyMinHeapError::AllocationFailed`] rather than aborting the
/// process.
pub const MAX_CAPACITY: usize = isize::MAX as usize / 64;

/// Error returned by [`LazyMinHeap::try_with_capacity`],
/// [`LazyMinHeap::try_reserve`], and [`LazyMinHeap::try_rebuild`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LazyMinHeapError {
    /// The requested capacity exceeds [`MAX_CAPACITY`].
    CapacityTooLarge {
        /// The capacity that was requested.
        requested: usize,
        /// The configured upper bound.
        max: usize,
    },
    /// The allocator could not satisfy the reservation for the
    /// requested capacity.
    ///
    /// Returned instead of aborting the process when `capacity *
    /// size_of::<_>()` exceeds what the allocator can provide
    /// (including the case where the byte count overflows
    /// `isize::MAX`).
    AllocationFailed {
        /// The capacity whose allocation failed.
        requested: usize,
    },
}

impl fmt::Display for LazyMinHeapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LazyMinHeapError::CapacityTooLarge { requested, max } => {
                write!(f, "LazyMinHeap capacity {requested} exceeds maximum {max}")
            },
            LazyMinHeapError::AllocationFailed { requested } => {
                write!(
                    f,
                    "LazyMinHeap failed to allocate backing storage for capacity {requested}"
                )
            },
        }
    }
}

impl std::error::Error for LazyMinHeapError {}

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
/// if let Some((victim, _count)) = freq.pop_best() {
///     assert!(victim == "page_b" || victim == "page_c");  // Both have count 1
/// }
/// ```
#[derive(Clone)]
pub struct LazyMinHeap<K, S> {
    scores: HashMap<K, ScoreEntry<S>>,
    heap: BinaryHeap<Reverse<HeapEntry<K, S>>>,
    seq: u64,
    /// When `Some(factor)`, every [`update`](Self::update) finishes
    /// with a [`maybe_rebuild`](Self::maybe_rebuild) call at the
    /// configured factor, bounding stale heap growth at roughly
    /// `len() * factor` entries.
    auto_rebuild: Option<usize>,
}

impl<K, S> fmt::Debug for LazyMinHeap<K, S> {
    /// Redacted `Debug` output.
    ///
    /// Historical derived `Debug` recursed through every `(key,
    /// score)` pair and every stale heap entry, which exposed all
    /// cache keys via `tracing::debug!`, `dbg!`, and panic
    /// backtraces. This impl deliberately does **not** require
    /// `K: Debug` / `S: Debug` and reports only aggregate counters.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LazyMinHeap")
            .field("len", &self.scores.len())
            .field("heap_len", &self.heap.len())
            .field("seq", &self.seq)
            .field("auto_rebuild_factor", &self.auto_rebuild)
            .finish_non_exhaustive()
    }
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
            auto_rebuild: None,
        }
    }

    /// Creates an empty heap with pre-allocated capacity.
    ///
    /// # Panics
    ///
    /// Panics if `capacity > MAX_CAPACITY`. Use
    /// [`try_with_capacity`](Self::try_with_capacity) for a fallible
    /// variant that surfaces allocator failure as a
    /// [`LazyMinHeapError`].
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::LazyMinHeap;
    ///
    /// let heap: LazyMinHeap<i32, i32> = LazyMinHeap::with_capacity(1000);
    /// assert!(heap.is_empty());
    /// ```
    #[track_caller]
    pub fn with_capacity(capacity: usize) -> Self {
        Self::try_with_capacity(capacity)
            .expect("LazyMinHeap::with_capacity: capacity exceeds MAX_CAPACITY")
    }

    /// Fallible [`with_capacity`](Self::with_capacity).
    ///
    /// Returns [`LazyMinHeapError::CapacityTooLarge`] when `capacity`
    /// exceeds [`MAX_CAPACITY`], or
    /// [`LazyMinHeapError::AllocationFailed`] when the allocator
    /// cannot satisfy the reservation.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::{LazyMinHeap, MAX_CAPACITY};
    ///
    /// let heap: LazyMinHeap<i32, i32> =
    ///     LazyMinHeap::try_with_capacity(1000).unwrap();
    /// assert!(heap.is_empty());
    ///
    /// assert!(LazyMinHeap::<u32, u32>::try_with_capacity(MAX_CAPACITY + 1).is_err());
    /// ```
    pub fn try_with_capacity(capacity: usize) -> Result<Self, LazyMinHeapError> {
        if capacity > MAX_CAPACITY {
            return Err(LazyMinHeapError::CapacityTooLarge {
                requested: capacity,
                max: MAX_CAPACITY,
            });
        }
        let mut scores: HashMap<K, ScoreEntry<S>> = HashMap::new();
        scores
            .try_reserve(capacity)
            .map_err(|_| LazyMinHeapError::AllocationFailed {
                requested: capacity,
            })?;
        let mut heap_vec: Vec<Reverse<HeapEntry<K, S>>> = Vec::new();
        heap_vec
            .try_reserve_exact(capacity)
            .map_err(|_| LazyMinHeapError::AllocationFailed {
                requested: capacity,
            })?;
        Ok(Self {
            scores,
            heap: BinaryHeap::from(heap_vec),
            seq: 0,
            auto_rebuild: None,
        })
    }

    /// Reserves capacity for at least `additional` more entries.
    ///
    /// # Panics
    ///
    /// Panics if the post-reservation capacity would exceed
    /// [`MAX_CAPACITY`] or if the allocator aborts. Use
    /// [`try_reserve`](Self::try_reserve) for a fallible variant.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::LazyMinHeap;
    ///
    /// let mut heap: LazyMinHeap<i32, i32> = LazyMinHeap::new();
    /// heap.reserve(100);
    /// ```
    #[track_caller]
    pub fn reserve(&mut self, additional: usize) {
        self.try_reserve(additional)
            .expect("LazyMinHeap::reserve: capacity exceeds MAX_CAPACITY");
    }

    /// Fallible [`reserve`](Self::reserve).
    ///
    /// Returns [`LazyMinHeapError::CapacityTooLarge`] when the
    /// requested total would exceed [`MAX_CAPACITY`], or
    /// [`LazyMinHeapError::AllocationFailed`] when the allocator
    /// cannot satisfy the reservation.
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), LazyMinHeapError> {
        let projected = self.scores.len().checked_add(additional).ok_or(
            LazyMinHeapError::CapacityTooLarge {
                requested: additional,
                max: MAX_CAPACITY,
            },
        )?;
        if projected > MAX_CAPACITY {
            return Err(LazyMinHeapError::CapacityTooLarge {
                requested: projected,
                max: MAX_CAPACITY,
            });
        }
        self.scores
            .try_reserve(additional)
            .map_err(|_| LazyMinHeapError::AllocationFailed {
                requested: additional,
            })?;
        // `BinaryHeap::try_reserve` is not stable. Reserve the
        // backing `Vec` indirectly by swapping the heap out into a
        // `Vec`, calling `try_reserve` on that, then swapping the
        // heap back in. Crucially, we swap the heap back even on
        // reservation failure so that partial allocator pressure
        // does not silently wipe the heap's contents.
        let mut vec: Vec<Reverse<HeapEntry<K, S>>> = std::mem::take(&mut self.heap).into_vec();
        let reserve_result = vec.try_reserve(additional);
        self.heap = BinaryHeap::from(vec);
        reserve_result.map_err(|_| LazyMinHeapError::AllocationFailed {
            requested: additional,
        })
    }

    /// Enables automatic [`maybe_rebuild`](Self::maybe_rebuild) on
    /// every [`update`](Self::update), bounding stale heap growth.
    ///
    /// `factor` follows the same convention as
    /// [`maybe_rebuild`](Self::maybe_rebuild): a rebuild triggers
    /// when `heap_len() > len() * factor`. Values below `1` are
    /// clamped to `1`. Pass this value to the builder to mitigate the
    /// unbounded-growth DoS described in the module-level **Security
    /// Considerations** section.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::LazyMinHeap;
    ///
    /// let mut heap: LazyMinHeap<&str, u32> =
    ///     LazyMinHeap::with_auto_rebuild(4);
    /// for i in 0..1_000 {
    ///     heap.update("hot", i);
    /// }
    /// // heap_len stays bounded rather than growing to 1_000.
    /// assert!(heap.heap_len() <= 4);
    /// ```
    pub fn with_auto_rebuild(factor: usize) -> Self {
        let mut heap = Self::new();
        heap.set_auto_rebuild(Some(factor));
        heap
    }

    /// Configures automatic rebuild on [`update`](Self::update).
    ///
    /// Passing `None` disables auto-rebuild (the default). Passing
    /// `Some(factor)` mirrors [`with_auto_rebuild`](Self::with_auto_rebuild).
    pub fn set_auto_rebuild(&mut self, factor: Option<usize>) {
        self.auto_rebuild = factor.map(|f| f.max(1));
    }

    /// Returns the currently configured auto-rebuild factor, if any.
    pub fn auto_rebuild_factor(&self) -> Option<usize> {
        self.auto_rebuild
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

    /// Clears all entries, retaining allocated capacity.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::LazyMinHeap;
    ///
    /// let mut heap: LazyMinHeap<&str, i32> = LazyMinHeap::with_capacity(100);
    /// heap.update("a", 1);
    /// heap.update("b", 2);
    ///
    /// heap.clear();
    /// assert!(heap.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.scores.clear();
        self.heap.clear();
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
        self.clear();
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

    /// Iterates over live `(key, score)` pairs in arbitrary order.
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
    /// let mut entries: Vec<_> = heap.iter().collect();
    /// entries.sort();
    /// assert_eq!(entries, vec![(&"a", &1), (&"b", &2)]);
    /// ```
    pub fn iter(&self) -> Iter<'_, K, S> {
        Iter {
            inner: self.scores.iter(),
        }
    }

    /// Returns the current score for `key`, if present.
    ///
    /// Accepts any borrowed form of the key type.
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
    pub fn score_of<Q>(&self, key: &Q) -> Option<&S>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
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
        // Guard against sequence-number wraparound. After ~2^64
        // updates the counter would otherwise wrap and a stale heap
        // entry could satisfy the `(score, seq)` equality check in
        // `pop_best`. Renumbering at the overflow boundary keeps the
        // check sound.
        if self.seq == u64::MAX {
            self.renumber_seqs();
        }
        let seq = self.seq;
        self.seq = self
            .seq
            .checked_add(1)
            .expect("LazyMinHeap::update: seq overflow after renumber (unreachable)");
        let previous = self.scores.insert(
            key.clone(),
            ScoreEntry {
                score: score.clone(),
                seq,
            },
        );
        self.push_entry_with_seq(key, score, seq);

        if let Some(factor) = self.auto_rebuild {
            self.maybe_rebuild(factor);
        }

        previous.map(|entry| entry.score)
    }

    /// Renumbers all live entries with fresh sequential seqs and
    /// resets the counter to `len()`.
    ///
    /// Preserves FIFO tie-breaking order by sorting live entries by
    /// their existing seq before renumbering. Called automatically on
    /// [`update`](Self::update) at the overflow boundary; exposed here
    /// so callers that retain heaps across extremely long-running
    /// processes can force the renumber explicitly.
    pub fn renumber_seqs(&mut self) {
        // Drain live entries preserving original seq order so that
        // equal-score keys keep their FIFO position post-renumber.
        let mut live: Vec<(K, S, u64)> = self
            .scores
            .iter()
            .map(|(k, entry)| (k.clone(), entry.score.clone(), entry.seq))
            .collect();
        live.sort_by_key(|(_, _, seq)| *seq);

        self.scores.clear();
        self.heap.clear();
        self.seq = 0;
        for (key, score, _) in live {
            let seq = self.seq;
            self.seq += 1;
            self.scores.insert(
                key.clone(),
                ScoreEntry {
                    score: score.clone(),
                    seq,
                },
            );
            self.push_entry_with_seq(key, score, seq);
        }
    }

    /// Removes `key` and returns its score, if present.
    ///
    /// Accepts any borrowed form of the key type. This only removes from
    /// the authoritative map; stale heap entries will be skipped by
    /// [`pop_best`](Self::pop_best).
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
    pub fn remove<Q>(&mut self, key: &Q) -> Option<S>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
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

    /// Returns references to the minimum `(key, score)` without removing it.
    ///
    /// Stale heap roots are discarded in place so the returned reference always
    /// points at a live entry. Takes `&mut self` for that reason — repeated
    /// calls without intervening updates are O(1).
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
    /// // peek does not consume.
    /// assert_eq!(heap.peek_best(), Some((&"high", &1)));
    /// assert_eq!(heap.peek_best(), Some((&"high", &1)));
    /// assert_eq!(heap.len(), 2);
    ///
    /// // pop_best returns the same entry peek_best showed.
    /// assert_eq!(heap.pop_best(), Some(("high", 1)));
    /// assert_eq!(heap.peek_best(), Some((&"low", &10)));
    /// ```
    pub fn peek_best(&mut self) -> Option<(&K, &S)> {
        loop {
            match self.heap.peek() {
                Some(Reverse(top)) => {
                    let live = self.scores.get(&top.key).is_some_and(|current| {
                        current.score == top.score && current.seq == top.seq
                    });
                    if live {
                        break;
                    }
                },
                None => return None,
            }
            self.heap.pop();
        }

        let Reverse(top) = self.heap.peek()?;
        self.scores
            .get_key_value(&top.key)
            .map(|(k, entry)| (k, &entry.score))
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

    /// Fallible [`rebuild`](Self::rebuild) that routes backing
    /// allocations through `Vec::try_reserve_exact`.
    ///
    /// Returns [`LazyMinHeapError::AllocationFailed`] without mutating
    /// the heap when the allocator cannot satisfy the new heap
    /// buffer. Prefer this over [`rebuild`](Self::rebuild) when the
    /// population size is attacker-influenced.
    pub fn try_rebuild(&mut self) -> Result<(), LazyMinHeapError> {
        let n = self.scores.len();
        let mut new_vec: Vec<Reverse<HeapEntry<K, S>>> = Vec::new();
        new_vec
            .try_reserve_exact(n)
            .map_err(|_| LazyMinHeapError::AllocationFailed { requested: n })?;
        for (key, entry) in self.scores.iter() {
            new_vec.push(Reverse(HeapEntry {
                score: entry.score.clone(),
                seq: entry.seq,
                key: key.clone(),
            }));
        }
        self.heap = BinaryHeap::from(new_vec);
        Ok(())
    }

    /// Rebuilds if the heap has grown too stale relative to map size.
    ///
    /// Triggers rebuild when `heap_len() > len() * factor`. Values of
    /// `factor` below 1 are clamped to 1.
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

    #[cfg(test)]
    fn debug_snapshot(&self) -> LazyHeapSnapshot {
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
        // Saturate on overflow. `capacity() * size_of::<_>()` can
        // overflow `usize` for pathologically large capacities;
        // without saturation this would panic in debug builds and
        // wrap silently in release.
        let scores_bytes = self
            .scores
            .capacity()
            .saturating_mul(std::mem::size_of::<(K, ScoreEntry<S>)>());
        let heap_bytes = self
            .heap
            .capacity()
            .saturating_mul(std::mem::size_of::<std::cmp::Reverse<HeapEntry<K, S>>>());
        std::mem::size_of::<Self>()
            .saturating_add(scores_bytes)
            .saturating_add(heap_bytes)
    }

    #[cfg(test)]
    fn debug_snapshot_scores(&self) -> Vec<(K, S)>
    where
        K: Clone,
        S: Clone,
    {
        self.scores
            .iter()
            .map(|(key, entry)| (key.clone(), entry.score.clone()))
            .collect()
    }

    #[cfg(test)]
    fn debug_validate_invariants(&self) {
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

/// Borrowing iterator over live `(key, score)` pairs.
///
/// Created by [`LazyMinHeap::iter`].
pub struct Iter<'a, K, S> {
    inner: std::collections::hash_map::Iter<'a, K, ScoreEntry<S>>,
}

impl<K: fmt::Debug, S: fmt::Debug> fmt::Debug for Iter<'_, K, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Iter").finish_non_exhaustive()
    }
}

impl<'a, K, S> Iterator for Iter<'a, K, S> {
    type Item = (&'a K, &'a S);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(key, entry)| (key, &entry.score))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<K, S> ExactSizeIterator for Iter<'_, K, S> {}
impl<K, S> FusedIterator for Iter<'_, K, S> {}

/// Owning iterator over live `(key, score)` pairs.
///
/// Created by the [`IntoIterator`] implementation on [`LazyMinHeap`].
pub struct IntoIter<K, S> {
    inner: std::collections::hash_map::IntoIter<K, ScoreEntry<S>>,
}

impl<K: fmt::Debug, S: fmt::Debug> fmt::Debug for IntoIter<K, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IntoIter").finish_non_exhaustive()
    }
}

impl<K, S> Iterator for IntoIter<K, S> {
    type Item = (K, S);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(key, entry)| (key, entry.score))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<K, S> ExactSizeIterator for IntoIter<K, S> {}
impl<K, S> FusedIterator for IntoIter<K, S> {}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct LazyHeapSnapshot {
    len: usize,
    heap_len: usize,
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

impl<K, S> FromIterator<(K, S)> for LazyMinHeap<K, S>
where
    K: Eq + Hash + Clone,
    S: Ord + Clone,
{
    fn from_iter<I: IntoIterator<Item = (K, S)>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        // Use the *fallible* `try_with_capacity` and silently fall
        // back to a zero-capacity heap when the caller-reported
        // `size_hint` cannot be honored. This prevents an oversized
        // or adversarial `size_hint` (e.g. `usize::MAX`) from turning
        // `FromIterator` into an allocator DoS / abort vector. See
        // the module-level **Security Considerations** section.
        let mut heap = Self::try_with_capacity(lower).unwrap_or_else(|_| Self::new());
        for (key, score) in iter {
            heap.update(key, score);
        }
        heap
    }
}

impl<K, S> Extend<(K, S)> for LazyMinHeap<K, S>
where
    K: Eq + Hash + Clone,
    S: Ord + Clone,
{
    fn extend<I: IntoIterator<Item = (K, S)>>(&mut self, iter: I) {
        for (key, score) in iter {
            self.update(key, score);
        }
    }
}

impl<K, S> IntoIterator for LazyMinHeap<K, S>
where
    K: Eq + Hash + Clone,
    S: Ord + Clone,
{
    type Item = (K, S);
    type IntoIter = IntoIter<K, S>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            inner: self.scores.into_iter(),
        }
    }
}

impl<'a, K, S> IntoIterator for &'a LazyMinHeap<K, S>
where
    K: Eq + Hash + Clone,
    S: Ord + Clone,
{
    type Item = (&'a K, &'a S);
    type IntoIter = Iter<'a, K, S>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
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
    fn peek_best_returns_min_without_removing() {
        let mut heap = LazyMinHeap::new();
        heap.update("a", 5);
        heap.update("b", 2);
        heap.update("c", 9);

        assert_eq!(heap.peek_best(), Some((&"b", &2)));
        assert_eq!(heap.peek_best(), Some((&"b", &2)));
        assert_eq!(heap.len(), 3);
    }

    #[test]
    fn peek_best_skips_stale_roots() {
        let mut heap = LazyMinHeap::new();
        heap.update("a", 1);
        heap.update("a", 5); // makes the (a, 1) entry stale
        heap.update("b", 3);

        // (a,1) was the old min but is stale; live min is (b, 3).
        assert_eq!(heap.peek_best(), Some((&"b", &3)));
        // Stale roots removed in place; subsequent pop matches.
        assert_eq!(heap.pop_best(), Some(("b", 3)));
        assert_eq!(heap.peek_best(), Some((&"a", &5)));
    }

    #[test]
    fn peek_best_returns_none_when_empty() {
        let mut heap: LazyMinHeap<&str, u32> = LazyMinHeap::new();
        assert_eq!(heap.peek_best(), None);

        heap.update("a", 1);
        assert!(heap.peek_best().is_some());
        heap.remove(&"a");
        assert_eq!(heap.peek_best(), None);
    }

    #[test]
    fn peek_best_matches_pop_best() {
        let mut heap = LazyMinHeap::new();
        for (k, s) in [("a", 5), ("b", 2), ("c", 9), ("d", 1), ("a", 3)] {
            heap.update(k, s);
        }
        // Iterate peek/pop pairs and confirm peek predicts the next pop result.
        let mut peeked = Vec::new();
        let mut popped = Vec::new();
        while let Some((k, s)) = heap.peek_best() {
            peeked.push((*k, *s));
            popped.push(heap.pop_best().unwrap());
        }
        assert_eq!(peeked, popped);
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

    #[test]
    fn lazy_heap_iter_visits_live_entries() {
        let mut heap = LazyMinHeap::new();
        heap.update("a", 3);
        heap.update("b", 1);
        heap.update("a", 2);

        let mut entries: Vec<_> = heap.iter().collect();
        entries.sort();
        assert_eq!(entries, vec![(&"a", &2), (&"b", &1)]);
    }

    #[test]
    fn lazy_heap_into_iter_yields_live_entries() {
        let mut heap = LazyMinHeap::new();
        heap.update("a", 3);
        heap.update("b", 1);
        heap.update("a", 2);

        let mut entries: Vec<_> = heap.into_iter().collect();
        entries.sort();
        assert_eq!(entries, vec![("a", 2), ("b", 1)]);
    }

    // =============================================================================
    // Security Regression Tests
    // =============================================================================

    #[test]
    fn try_with_capacity_rejects_oversized_capacity() {
        let err = LazyMinHeap::<u32, u32>::try_with_capacity(MAX_CAPACITY + 1).unwrap_err();
        assert!(matches!(err, LazyMinHeapError::CapacityTooLarge { .. }));
    }

    #[test]
    #[should_panic(expected = "MAX_CAPACITY")]
    fn with_capacity_panics_on_oversized_capacity() {
        let _heap: LazyMinHeap<u32, u32> = LazyMinHeap::with_capacity(MAX_CAPACITY + 1);
    }

    #[test]
    fn try_reserve_rejects_oversized_total() {
        let mut heap: LazyMinHeap<u32, u32> = LazyMinHeap::new();
        let err = heap.try_reserve(MAX_CAPACITY + 1).unwrap_err();
        assert!(matches!(err, LazyMinHeapError::CapacityTooLarge { .. }));
    }

    #[test]
    fn try_reserve_small_value_succeeds_and_is_usable() {
        let mut heap: LazyMinHeap<u32, u32> = LazyMinHeap::new();
        heap.try_reserve(16).expect("reserve should succeed");
        heap.update(1, 10);
        heap.update(2, 5);
        assert_eq!(heap.pop_best(), Some((2, 5)));
        assert_eq!(heap.pop_best(), Some((1, 10)));
    }

    #[test]
    fn try_rebuild_preserves_live_entries_and_removes_stale() {
        let mut heap: LazyMinHeap<&str, u32> = LazyMinHeap::new();
        heap.update("a", 3);
        heap.update("a", 2);
        heap.update("a", 1);
        heap.update("b", 4);
        assert!(heap.heap_len() > heap.len());

        heap.try_rebuild().expect("rebuild should succeed");
        assert_eq!(heap.heap_len(), heap.len());
        assert_eq!(heap.pop_best(), Some(("a", 1)));
        assert_eq!(heap.pop_best(), Some(("b", 4)));
    }

    #[test]
    fn auto_rebuild_bounds_stale_heap_growth() {
        // Without auto_rebuild: heap_len grows unbounded.
        let mut heap: LazyMinHeap<&str, u32> = LazyMinHeap::new();
        for i in 0..1_000 {
            heap.update("hot", i);
        }
        assert_eq!(heap.len(), 1);
        assert_eq!(heap.heap_len(), 1_000);

        // With auto_rebuild(4): heap_len stays bounded.
        let mut heap: LazyMinHeap<&str, u32> = LazyMinHeap::with_auto_rebuild(4);
        for i in 0..1_000 {
            heap.update("hot", i);
        }
        assert_eq!(heap.len(), 1);
        assert!(heap.heap_len() <= 4);
        assert_eq!(heap.auto_rebuild_factor(), Some(4));
    }

    #[test]
    fn set_auto_rebuild_clamps_factor_below_one() {
        let mut heap: LazyMinHeap<&str, u32> = LazyMinHeap::new();
        heap.set_auto_rebuild(Some(0));
        assert_eq!(heap.auto_rebuild_factor(), Some(1));
        heap.set_auto_rebuild(None);
        assert_eq!(heap.auto_rebuild_factor(), None);
    }

    #[test]
    fn renumber_seqs_preserves_fifo_for_equal_scores() {
        let mut heap: LazyMinHeap<&str, u32> = LazyMinHeap::new();
        heap.update("first", 1);
        heap.update("second", 1);
        heap.update("third", 1);
        heap.renumber_seqs();
        // Pop order must still reflect insertion order.
        assert_eq!(heap.pop_best(), Some(("first", 1)));
        assert_eq!(heap.pop_best(), Some(("second", 1)));
        assert_eq!(heap.pop_best(), Some(("third", 1)));
    }

    #[test]
    fn update_at_seq_saturation_renumbers_without_stale_match() {
        // Simulate a heap about to wrap its sequence counter.
        let mut heap: LazyMinHeap<&str, u32> = LazyMinHeap::new();
        heap.update("a", 1);
        heap.update("b", 2);

        // Fast-forward seq to the overflow boundary. The next `update`
        // must renumber rather than wrap the counter.
        heap.seq = u64::MAX;

        heap.update("c", 3);

        // All three keys present, correct order, fresh seqs.
        assert_eq!(heap.len(), 3);
        assert_eq!(heap.pop_best(), Some(("a", 1)));
        assert_eq!(heap.pop_best(), Some(("b", 2)));
        assert_eq!(heap.pop_best(), Some(("c", 3)));
        assert!(heap.seq < u64::MAX);
    }

    #[test]
    fn approx_bytes_does_not_overflow() {
        let heap: LazyMinHeap<u64, u64> = LazyMinHeap::with_capacity(1_024);
        // Saturating arithmetic: result is well-defined and finite.
        let bytes = heap.approx_bytes();
        assert!(bytes >= std::mem::size_of::<LazyMinHeap<u64, u64>>());
        assert!(bytes < usize::MAX);
    }

    #[test]
    fn debug_impl_does_not_leak_keys_or_scores() {
        // Keys and scores intentionally do *not* implement Debug; a
        // derived Debug would fail to compile, proving the redacted
        // impl avoids exposing them.
        struct Secret(u32);
        impl PartialEq for Secret {
            fn eq(&self, other: &Self) -> bool {
                self.0 == other.0
            }
        }
        impl Eq for Secret {}
        impl Clone for Secret {
            fn clone(&self) -> Self {
                Secret(self.0)
            }
        }
        impl std::hash::Hash for Secret {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                self.0.hash(state)
            }
        }
        impl Ord for Secret {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.0.cmp(&other.0)
            }
        }
        impl PartialOrd for Secret {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        let mut heap: LazyMinHeap<Secret, Secret> = LazyMinHeap::new();
        heap.update(Secret(0xdead_beef), Secret(0xfeed_face));

        let rendered = format!("{:?}", heap);
        assert!(rendered.contains("LazyMinHeap"));
        assert!(rendered.contains("len"));
        assert!(!rendered.contains("dead"));
        assert!(!rendered.contains("beef"));
        assert!(!rendered.contains("feed"));
        assert!(!rendered.contains("face"));
    }

    #[test]
    fn from_iter_clamps_size_hint_to_max_capacity() {
        struct AdversarialHint;
        impl Iterator for AdversarialHint {
            type Item = (u32, u32);
            fn next(&mut self) -> Option<Self::Item> {
                None
            }
            fn size_hint(&self) -> (usize, Option<usize>) {
                (usize::MAX, None)
            }
        }
        // Must not panic or abort on the allocator.
        let heap: LazyMinHeap<u32, u32> = AdversarialHint.collect();
        assert!(heap.is_empty());
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
