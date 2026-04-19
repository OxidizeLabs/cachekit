//! Fixed-size access history ring buffer.
//!
//! Stores the last `K` timestamps in a ring buffer, providing O(1) record
//! and O(1) access to the k-th most recent entry. Essential for LRU-K policies
//! where eviction decisions depend on the K-th most recent access time.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                      FixedHistory<K=4> Layout                               │
//! │                                                                             │
//! │   Ring Buffer                                                               │
//! │   ────────────                                                              │
//! │                                                                             │
//! │   data: [u64; K]              cursor: next write position                   │
//! │   len: valid entries          (wraps around when full)                      │
//! │                                                                             │
//! │   After recording: 10, 20, 30, 40, 50                                       │
//! │                                                                             │
//! │   Index:     0     1     2     3                                            │
//! │            ┌─────┬─────┬─────┬─────┐                                        │
//! │   data:    │ 50  │ 20  │ 30  │ 40  │                                        │
//! │            └─────┴─────┴─────┴─────┘                                        │
//! │              ▲                                                              │
//! │              │                                                              │
//! │           cursor = 1 (next write goes here)                                 │
//! │                                                                             │
//! │   Access Pattern                                                            │
//! │   ──────────────                                                            │
//! │                                                                             │
//! │   kth_most_recent(k) = data[(cursor + K - k) % K]                           │
//! │                                                                             │
//! │   k=1 (most recent):  data[(1 + 4 - 1) % 4] = data[0] = 50                  │
//! │   k=2:                data[(1 + 4 - 2) % 4] = data[3] = 40                  │
//! │   k=3:                data[(1 + 4 - 3) % 4] = data[2] = 30                  │
//! │   k=4 (oldest):       data[(1 + 4 - 4) % 4] = data[1] = 20                  │
//! │                                                                             │
//! │   Record Flow                                                               │
//! │   ───────────                                                               │
//! │                                                                             │
//! │   record(60):                                                               │
//! │     1. data[cursor] = 60         → data[1] = 60                             │
//! │     2. cursor = (cursor + 1) % K → cursor = 2                               │
//! │     3. len stays at K (already full)                                        │
//! │                                                                             │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Key Components
//!
//! - [`FixedHistory`]: Fixed-size ring buffer for timestamp history
//! - [`Iter`]: Borrowed iterator over timestamps in MRU order
//! - [`IntoIter`]: Owning iterator over timestamps in MRU order
//!
//! ## Operations
//!
//! | Operation             | Description                      | Complexity |
//! |-----------------------|----------------------------------|------------|
//! | [`record`]            | Add timestamp (overwrites oldest)| O(1)       |
//! | [`most_recent`]       | Get most recent timestamp        | O(1)       |
//! | [`kth_most_recent`]   | Get k-th most recent timestamp   | O(1)       |
//! | [`iter`] / [`into_iter`] | Iterate in MRU order          | O(K)       |
//! | [`to_vec_mru`]        | Collect all into a Vec (MRU)     | O(K)       |
//!
//! [`record`]: FixedHistory::record
//! [`most_recent`]: FixedHistory::most_recent
//! [`kth_most_recent`]: FixedHistory::kth_most_recent
//! [`iter`]: FixedHistory::iter
//! [`into_iter`]: FixedHistory#impl-IntoIterator
//! [`to_vec_mru`]: FixedHistory::to_vec_mru
//!
//! ## Use Cases
//!
//! - **LRU-K policy**: Track last K access times per entry for eviction decisions
//! - **Access frequency**: Count accesses within a time window
//! - **Temporal patterns**: Detect periodic access patterns
//!
//! ## Example Usage
//!
//! ```
//! use cachekit::ds::FixedHistory;
//!
//! // Track last 3 access times
//! let mut history = FixedHistory::<3>::new();
//!
//! // Record access timestamps
//! history.record(100);
//! history.record(200);
//! history.record(300);
//!
//! // Most recent access
//! assert_eq!(history.most_recent(), Some(300));
//!
//! // LRU-K: Get K-th most recent for eviction comparison
//! assert_eq!(history.kth_most_recent(3), Some(100));  // Oldest of K
//!
//! // Overwrites oldest when full
//! history.record(400);
//! assert_eq!(history.to_vec_mru(), vec![400, 300, 200]);  // 100 is gone
//! ```
//!
//! ## Use Case: LRU-2 Eviction
//!
//! ```
//! use cachekit::ds::FixedHistory;
//!
//! // LRU-2: Evict based on 2nd most recent access time
//! struct CacheEntry {
//!     value: String,
//!     history: FixedHistory<2>,
//! }
//!
//! impl CacheEntry {
//!     fn new(value: String, timestamp: u64) -> Self {
//!         let mut entry = CacheEntry {
//!             value,
//!             history: FixedHistory::new(),
//!         };
//!         entry.history.record(timestamp);
//!         entry
//!     }
//!
//!     fn access(&mut self, timestamp: u64) {
//!         self.history.record(timestamp);
//!     }
//!
//!     // LRU-2 uses the 2nd most recent access for comparison
//!     fn eviction_priority(&self) -> u64 {
//!         // If only 1 access, use that; otherwise use 2nd most recent
//!         self.history.kth_most_recent(2)
//!             .or(self.history.most_recent())
//!             .unwrap_or(0)
//!     }
//! }
//!
//! let mut entry = CacheEntry::new("data".into(), 100);
//! assert_eq!(entry.eviction_priority(), 100);  // Only 1 access
//!
//! entry.access(200);
//! assert_eq!(entry.eviction_priority(), 100);  // 2nd most recent = 100
//!
//! entry.access(300);
//! assert_eq!(entry.eviction_priority(), 200);  // 2nd most recent = 200
//! ```
//!
//! ## Thread Safety
//!
//! `FixedHistory` is not thread-safe. It is typically embedded within
//! cache entries and protected by the cache's synchronization.
//!
//! ## Security & Invariants
//!
//! - **Trusted timestamps.** The caller is responsible for supplying
//!   monotonically non-decreasing timestamps drawn from a trusted source
//!   (e.g. a coarse monotonic counter). If an adversary can influence the
//!   timestamp stream — for example via a wall-clock that can jump
//!   backwards, or a user-controlled counter — they can manipulate
//!   LRU-K-style eviction decisions built on top of this type. See
//!   [`FixedHistory::record`] for the full trust boundary.
//! - **Capacity bound.** `K` is capped at [`MAX_K`] (see [`FixedHistory::new`]).
//!   This keeps the inline `[u64; K]` array from triggering stack exhaustion
//!   if `K` is ever influenced by code an attacker controls. For large `K`
//!   on restricted stacks, use [`FixedHistory::boxed`] to heap-allocate
//!   without materialising the array on the stack.
//! - **No stale-slot disclosure.** [`FixedHistory::clear`] zeroes the
//!   backing array in addition to resetting the cursor and length, and the
//!   manual `Debug` impl only prints logically-live entries in MRU order.
//!   Overwritten / cleared timestamps are not observable through the public
//!   API, `Debug` output, or panic messages.
//! - **Not constant-time.** [`PartialEq`] and [`std::hash::Hash`] iterate
//!   only as far as the shared prefix, so they are not suitable for
//!   comparing data that must remain secret. `FixedHistory` is intended for
//!   access-timestamp bookkeeping, not for security-sensitive values.
//!
//! ## Implementation Notes
//!
//! - Uses a fixed-size inline array by default (no heap allocation);
//!   [`FixedHistory::boxed`] is provided for heap-allocating large `K`.
//! - Const generic `K` determines history depth at compile time
//! - Zero-size history (`K=0`) is a no-op
//! - Construction is bounded by [`MAX_K`]; exceeding it is a compile-time error
//! - `debug_validate_invariants()` available in debug/test builds
//!
//! [`MAX_K`]: MAX_K

/// Upper bound on `K` for [`FixedHistory`].
///
/// `FixedHistory<K>` stores `[u64; K]` inline, so an arbitrary `K` would risk
/// stack exhaustion (especially since [`FixedHistory::new`] returns by value
/// and materialises the array on the stack before any move).
///
/// This limit is enforced at compile time by [`FixedHistory::new`] and is
/// deliberately generous: realistic LRU-K policies use `K` in the single
/// digits, so `4096` leaves multiple orders of magnitude of headroom while
/// keeping the worst-case stack footprint at 32 KiB.
pub const MAX_K: usize = 4096;

/// Fixed-size ring buffer of the last `K` timestamps.
///
/// Stores timestamps in a circular buffer, automatically overwriting the oldest
/// entry when full. Provides O(1) access to any of the last K timestamps.
///
/// Implements [`Clone`], [`Copy`], [`Debug`], [`PartialEq`], [`Eq`], [`Hash`],
/// and [`IntoIterator`]. See [`iter`](Self::iter) for borrowed iteration in MRU order.
///
/// # Type Parameters
///
/// - `K`: Maximum number of timestamps to retain (const generic)
///
/// # Example
///
/// ```
/// use cachekit::ds::FixedHistory;
///
/// let mut history = FixedHistory::<3>::new();
///
/// history.record(10);
/// history.record(20);
/// history.record(30);
///
/// assert_eq!(history.most_recent(), Some(30));
/// assert_eq!(history.kth_most_recent(2), Some(20));
/// assert_eq!(history.kth_most_recent(3), Some(10));
///
/// // When full, oldest is overwritten
/// history.record(40);
/// assert_eq!(history.to_vec_mru(), vec![40, 30, 20]);
/// ```
///
/// # Use Case: Access Frequency Window
///
/// ```
/// use cachekit::ds::FixedHistory;
///
/// // Track last 5 access times
/// let mut history = FixedHistory::<5>::new();
///
/// // Simulate accesses at various times
/// for ts in [100, 150, 180, 200, 250] {
///     history.record(ts);
/// }
///
/// // Check if accessed recently (within last 100 time units)
/// let now = 260;
/// let oldest_in_window = history.kth_most_recent(5).unwrap();
/// let window_duration = now - oldest_in_window;
///
/// assert_eq!(window_duration, 160);  // 5 accesses over 160 time units
/// ```
#[derive(Clone, Copy)]
pub struct FixedHistory<const K: usize> {
    data: [u64; K],
    len: usize,
    cursor: usize,
}

// Manual `Debug` impl: only print the logically-live entries in MRU order,
// not the raw ring buffer. This prevents stale slots (left behind after a
// `clear()` or a wrap) from appearing in panic messages / logs, which would
// otherwise leak past access patterns that the public API already hides.
impl<const K: usize> std::fmt::Debug for FixedHistory<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FixedHistory")
            .field("capacity", &K)
            .field("len", &self.len)
            .field("mru", &self.to_vec_mru())
            .finish()
    }
}

impl<const K: usize> FixedHistory<K> {
    /// Creates an empty history.
    ///
    /// # Compile-time bound
    ///
    /// `K` must be less than or equal to [`MAX_K`]. Instantiating
    /// `FixedHistory<K>` with a larger `K` fails compilation. This prevents
    /// stack exhaustion from pathological `K` values and keeps the inline
    /// `[u64; K]` array to a bounded size.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FixedHistory;
    ///
    /// let history = FixedHistory::<4>::new();
    /// assert!(history.is_empty());
    /// assert_eq!(history.capacity(), 4);
    /// ```
    pub fn new() -> Self {
        const {
            assert!(
                K <= MAX_K,
                "FixedHistory<K>: K exceeds MAX_K (see cachekit::ds::fixed_history::MAX_K)"
            );
        }
        Self {
            data: [0; K],
            len: 0,
            cursor: 0,
        }
    }

    /// Creates an empty history on the heap without materialising the
    /// `[u64; K]` array on the stack.
    ///
    /// For small `K` this is equivalent to `Box::new(Self::new())` and
    /// slightly slower. For large `K` (up to [`MAX_K`]) the difference is
    /// significant: `Self::new()` returns `Self` by value, and a naive
    /// `Box::new(Self::new())` may materialise the 32 KiB array on the
    /// stack before moving it to the heap. That stack copy is an
    /// availability concern on restricted stacks (async tasks on small
    /// runtimes, threads created with a custom stack size, `no_std`
    /// targets). `boxed()` sidesteps this by zero-initialising the heap
    /// allocation in place.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FixedHistory;
    ///
    /// let mut history: Box<FixedHistory<4096>> = FixedHistory::boxed();
    /// history.record(100);
    /// assert_eq!(history.most_recent(), Some(100));
    /// ```
    pub fn boxed() -> Box<Self> {
        const {
            assert!(
                K <= MAX_K,
                "FixedHistory<K>: K exceeds MAX_K (see cachekit::ds::fixed_history::MAX_K)"
            );
        }
        // `FixedHistory` is `[u64; K]` + two `usize`s. Its canonical empty
        // state has every byte set to zero (`data = [0; K]`, `len = 0`,
        // `cursor = 0`), and it contains no padding, pointers, references,
        // or other types for which the all-zero bit pattern is invalid.
        // Writing zeros into a heap-allocated `MaybeUninit<Self>` therefore
        // produces a valid `Self` without ever touching the stack.
        let mut uninit: Box<std::mem::MaybeUninit<Self>> = Box::new_uninit();
        // SAFETY: `uninit` points to `size_of::<Self>()` bytes of
        // heap-allocated storage. Writing zeros into all of them yields a
        // valid `Self` for the reasons above, so `assume_init` is sound.
        unsafe {
            std::ptr::write_bytes(uninit.as_mut_ptr(), 0, 1);
            uninit.assume_init()
        }
    }

    /// Returns the maximum number of timestamps retained.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FixedHistory;
    ///
    /// let history = FixedHistory::<5>::new();
    /// assert_eq!(history.capacity(), 5);
    /// ```
    pub fn capacity(&self) -> usize {
        K
    }

    /// Returns the number of timestamps currently stored (<= `K`).
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FixedHistory;
    ///
    /// let mut history = FixedHistory::<3>::new();
    /// assert_eq!(history.len(), 0);
    ///
    /// history.record(100);
    /// history.record(200);
    /// assert_eq!(history.len(), 2);
    ///
    /// // Length caps at K
    /// history.record(300);
    /// history.record(400);
    /// assert_eq!(history.len(), 3);  // Still 3, oldest was overwritten
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if there are no timestamps recorded.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FixedHistory;
    ///
    /// let mut history = FixedHistory::<3>::new();
    /// assert!(history.is_empty());
    ///
    /// history.record(100);
    /// assert!(!history.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Records a timestamp, overwriting the oldest if the history is full.
    ///
    /// # Trust boundary
    ///
    /// `record` does **not** validate `timestamp` — it accepts any `u64`
    /// and treats the most recently recorded value as "now" for the
    /// purposes of LRU-K ordering. Callers are responsible for supplying
    /// timestamps from a trusted source, typically a monotonic counter
    /// incremented on each cache access. If an attacker can influence the
    /// timestamp stream (for example, because it is derived from a
    /// wall-clock that can jump, a user-supplied header, or any
    /// value not under the cache's exclusive control) they can:
    ///
    /// - Pin victim entries in cache forever by recording a far-future
    ///   timestamp, triggering eviction of legitimate entries (cache
    ///   pollution / DoS).
    /// - Make victim entries look "cold" by recording stale timestamps,
    ///   evicting hot entries they do not own (targeted eviction / cache
    ///   side channel).
    ///
    /// Store this value internally; never route it directly from network
    /// input.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FixedHistory;
    ///
    /// let mut history = FixedHistory::<2>::new();
    ///
    /// history.record(10);
    /// history.record(20);
    /// assert_eq!(history.to_vec_mru(), vec![20, 10]);
    ///
    /// // Overwrites oldest (10)
    /// history.record(30);
    /// assert_eq!(history.to_vec_mru(), vec![30, 20]);
    /// ```
    pub fn record(&mut self, timestamp: u64) {
        if K == 0 {
            return;
        }
        self.data[self.cursor] = timestamp;
        self.cursor = (self.cursor + 1) % K;
        if self.len < K {
            self.len += 1;
        }
    }

    /// Returns the most recently recorded timestamp.
    ///
    /// Returns `None` if the history is empty.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FixedHistory;
    ///
    /// let mut history = FixedHistory::<3>::new();
    /// assert_eq!(history.most_recent(), None);
    ///
    /// history.record(100);
    /// history.record(200);
    /// assert_eq!(history.most_recent(), Some(200));
    /// ```
    pub fn most_recent(&self) -> Option<u64> {
        self.kth_most_recent(1)
    }

    /// Returns the k-th most recent timestamp (`k = 1` is most recent).
    ///
    /// Returns `None` if `k` is 0, exceeds the number of recorded timestamps,
    /// or if the history is empty.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FixedHistory;
    ///
    /// let mut history = FixedHistory::<4>::new();
    /// history.record(10);
    /// history.record(20);
    /// history.record(30);
    ///
    /// // k=1 is most recent, k=3 is oldest
    /// assert_eq!(history.kth_most_recent(1), Some(30));
    /// assert_eq!(history.kth_most_recent(2), Some(20));
    /// assert_eq!(history.kth_most_recent(3), Some(10));
    ///
    /// // Out of bounds
    /// assert_eq!(history.kth_most_recent(0), None);
    /// assert_eq!(history.kth_most_recent(4), None);  // Only 3 recorded
    /// ```
    pub fn kth_most_recent(&self, k: usize) -> Option<u64> {
        if K == 0 || k == 0 || k > self.len {
            return None;
        }
        // Parenthesise as `cursor + (K - k)` so the intermediate stays below
        // `2 * K` even for the largest permitted `K`. `k <= self.len <= K`
        // guarantees `K - k` does not underflow.
        let idx = (self.cursor + (K - k)) % K;
        Some(self.data[idx])
    }

    /// Returns timestamps from most-recent to least-recent.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FixedHistory;
    ///
    /// let mut history = FixedHistory::<3>::new();
    /// history.record(100);
    /// history.record(200);
    /// history.record(300);
    ///
    /// assert_eq!(history.to_vec_mru(), vec![300, 200, 100]);
    ///
    /// // After wrap
    /// history.record(400);
    /// assert_eq!(history.to_vec_mru(), vec![400, 300, 200]);
    /// ```
    pub fn to_vec_mru(&self) -> Vec<u64> {
        (1..=self.len)
            .filter_map(|k| self.kth_most_recent(k))
            .collect()
    }

    /// Returns an iterator over recorded timestamps in MRU order (most recent first).
    ///
    /// Does **not** consume or modify the history.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FixedHistory;
    ///
    /// let mut history = FixedHistory::<4>::new();
    /// history.record(10);
    /// history.record(20);
    /// history.record(30);
    ///
    /// let timestamps: Vec<_> = history.iter().collect();
    /// assert_eq!(timestamps, vec![30, 20, 10]);
    /// ```
    pub fn iter(&self) -> Iter<'_, K> {
        Iter {
            history: self,
            pos: 1,
        }
    }

    /// Clears the history and resets cursor/length.
    ///
    /// Also zeroes the backing array so that previously recorded timestamps
    /// cannot be observed through `Debug` output, core dumps, or accidental
    /// raw-byte access. This is a minor hardening measure — the public API
    /// already respects `len` and never returns stale slots — but it removes
    /// a class of passive information-disclosure hazards.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FixedHistory;
    ///
    /// let mut history = FixedHistory::<3>::new();
    /// history.record(10);
    /// history.record(20);
    ///
    /// history.clear();
    /// assert!(history.is_empty());
    /// assert_eq!(history.most_recent(), None);
    /// ```
    pub fn clear(&mut self) {
        self.data = [0; K];
        self.len = 0;
        self.cursor = 0;
    }

    /// Clears the history (no heap allocations to shrink).
    ///
    /// Equivalent to [`clear`](Self::clear) since `FixedHistory` uses
    /// a fixed-size array with no heap allocation.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FixedHistory;
    ///
    /// let mut history = FixedHistory::<3>::new();
    /// history.record(10);
    ///
    /// history.clear_shrink();
    /// assert!(history.is_empty());
    /// ```
    pub fn clear_shrink(&mut self) {
        self.clear();
    }

    /// Returns an approximate memory footprint in bytes.
    ///
    /// Since `FixedHistory` uses a fixed-size array, this is constant
    /// regardless of how many timestamps are recorded.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FixedHistory;
    ///
    /// let history = FixedHistory::<10>::new();
    /// let bytes = history.approx_bytes();
    ///
    /// // Includes array of 10 u64s plus len and cursor
    /// assert!(bytes >= 10 * std::mem::size_of::<u64>());
    /// ```
    pub fn approx_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
    }

    /// Returns a debug snapshot of the history in MRU order.
    ///
    /// Only available in `cfg(test)` or `debug_assertions` builds. Not part
    /// of the stable API; do not rely on this being callable from release
    /// builds of downstream crates.
    #[cfg(any(test, debug_assertions))]
    #[doc(hidden)]
    pub fn debug_snapshot_mru(&self) -> Vec<u64> {
        self.to_vec_mru()
    }

    /// Asserts internal invariants; panics on violation.
    ///
    /// Only available in `cfg(test)` or `debug_assertions` builds. Not part
    /// of the stable API.
    #[cfg(any(test, debug_assertions))]
    #[doc(hidden)]
    pub fn debug_validate_invariants(&self) {
        assert!(self.len <= K);
        if K == 0 {
            assert_eq!(self.len, 0);
            assert_eq!(self.cursor, 0);
        } else {
            assert!(self.cursor < K);
        }
    }
}

impl<const K: usize> Default for FixedHistory<K> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// PartialEq, Eq, Hash — compare logical content, not the raw backing array
// (raw derive would flag stale slots as differences)
// ---------------------------------------------------------------------------

/// Compares logical content in MRU order, ignoring stale slots in the
/// backing array.
///
/// # Security
///
/// This implementation is **not constant-time**: it short-circuits on the
/// first differing MRU entry, leaking the length of the matching prefix
/// through timing. `FixedHistory` is intended for access-timestamp
/// bookkeeping and is not suitable for comparing values that must remain
/// secret. Do not use it as a `HashMap` key whose timestamps are derived
/// from untrusted input without a DoS-resistant hasher (see `Hash` below).
impl<const K: usize> PartialEq for FixedHistory<K> {
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }
        for k in 1..=self.len {
            if self.kth_most_recent(k) != other.kth_most_recent(k) {
                return false;
            }
        }
        true
    }
}

impl<const K: usize> Eq for FixedHistory<K> {}

/// Hashes logical content in MRU order so histories with the same timestamps
/// but different internal cursor positions hash equal (consistent with
/// [`PartialEq`]).
///
/// # Security
///
/// The standard-library default hasher is randomised and DoS-resistant,
/// but this trait will cooperate with any hasher chosen by the caller.
/// If `FixedHistory` values are used as keys in a map where timestamps
/// can be influenced by an adversary (for example, wall-clock readings
/// from untrusted input), pair this type with a DoS-resistant hasher
/// rather than a fast non-cryptographic one. The contents themselves are
/// not secret-equivalent — do not rely on hashing to hide timestamps.
impl<const K: usize> std::hash::Hash for FixedHistory<K> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.len.hash(state);
        for k in 1..=self.len {
            self.kth_most_recent(k).hash(state);
        }
    }
}

// ---------------------------------------------------------------------------
// Iterator types (C-ITER-TY: names match the methods that produce them)
// ---------------------------------------------------------------------------

/// Borrowed iterator over timestamps in a [`FixedHistory`], from most recent to oldest.
///
/// Created by [`FixedHistory::iter`].
#[derive(Debug, Clone)]
pub struct Iter<'a, const K: usize> {
    history: &'a FixedHistory<K>,
    pos: usize, // 1-indexed: 1 = most recent, history.len() = oldest
}

impl<'a, const K: usize> Iterator for Iter<'a, K> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        // The `?` must come *before* the increment: once the iterator is
        // exhausted (pos > len), `kth_most_recent` returns `None` and we
        // return without touching `pos`. This keeps `pos` bounded by
        // `len + 1 <= K + 1 <= MAX_K + 1`, so the `+= 1` below cannot
        // overflow no matter how many times `next()` is called after
        // exhaustion. Do not reorder.
        let val = self.history.kth_most_recent(self.pos)?;
        self.pos += 1;
        Some(val)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.history.len().saturating_sub(self.pos - 1);
        (remaining, Some(remaining))
    }
}

impl<const K: usize> ExactSizeIterator for Iter<'_, K> {}

/// Owning iterator over timestamps in a [`FixedHistory`], from most recent to oldest.
///
/// Created by calling [`IntoIterator::into_iter`] on a `FixedHistory`.
#[derive(Debug, Clone)]
pub struct IntoIter<const K: usize> {
    history: FixedHistory<K>,
    pos: usize,
}

impl<const K: usize> Iterator for IntoIter<K> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        // See `Iter::next` for the invariant justifying `+= 1`: the `?`
        // must remain before the increment so `pos` stays bounded once
        // exhausted.
        let val = self.history.kth_most_recent(self.pos)?;
        self.pos += 1;
        Some(val)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.history.len().saturating_sub(self.pos - 1);
        (remaining, Some(remaining))
    }
}

impl<const K: usize> ExactSizeIterator for IntoIter<K> {}

// ---------------------------------------------------------------------------
// IntoIterator impls (C-ITER: iter, into_iter)
// ---------------------------------------------------------------------------

impl<const K: usize> IntoIterator for FixedHistory<K> {
    type Item = u64;
    type IntoIter = IntoIter<K>;

    /// Consumes the history, returning an iterator over timestamps in MRU order.
    ///
    /// # Example
    ///
    /// ```
    /// use cachekit::ds::FixedHistory;
    ///
    /// let mut history = FixedHistory::<3>::new();
    /// history.record(10);
    /// history.record(20);
    ///
    /// let timestamps: Vec<_> = history.into_iter().collect();
    /// assert_eq!(timestamps, vec![20, 10]);
    /// ```
    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            history: self,
            pos: 1,
        }
    }
}

impl<'a, const K: usize> IntoIterator for &'a FixedHistory<K> {
    type Item = u64;
    type IntoIter = Iter<'a, K>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_history_tracks_last_k() {
        let mut history = FixedHistory::<3>::new();
        history.record(10);
        history.record(20);
        history.record(30);
        assert_eq!(history.to_vec_mru(), vec![30, 20, 10]);

        history.record(40);
        assert_eq!(history.to_vec_mru(), vec![40, 30, 20]);
        assert_eq!(history.kth_most_recent(3), Some(20));
    }

    #[test]
    fn fixed_history_empty_returns_none() {
        let history = FixedHistory::<4>::new();
        assert!(history.is_empty());
        assert_eq!(history.len(), 0);
        assert_eq!(history.most_recent(), None);
        assert_eq!(history.kth_most_recent(1), None);
        assert_eq!(history.to_vec_mru(), Vec::<u64>::new());
    }

    #[test]
    fn fixed_history_kth_bounds() {
        let mut history = FixedHistory::<3>::new();
        history.record(10);
        history.record(20);
        assert_eq!(history.kth_most_recent(0), None);
        assert_eq!(history.kth_most_recent(3), None);
        assert_eq!(history.kth_most_recent(1), Some(20));
        assert_eq!(history.kth_most_recent(2), Some(10));
    }

    #[test]
    fn fixed_history_wraps_and_overwrites_oldest() {
        let mut history = FixedHistory::<2>::new();
        history.record(1);
        history.record(2);
        assert_eq!(history.to_vec_mru(), vec![2, 1]);

        history.record(3);
        assert_eq!(history.to_vec_mru(), vec![3, 2]);
        assert_eq!(history.most_recent(), Some(3));
        assert_eq!(history.kth_most_recent(2), Some(2));
    }

    #[test]
    fn fixed_history_preserves_order_after_multiple_wraps() {
        let mut history = FixedHistory::<3>::new();
        for t in 1..=6 {
            history.record(t);
        }
        assert_eq!(history.len(), 3);
        assert_eq!(history.to_vec_mru(), vec![6, 5, 4]);
        assert_eq!(history.kth_most_recent(1), Some(6));
        assert_eq!(history.kth_most_recent(2), Some(5));
        assert_eq!(history.kth_most_recent(3), Some(4));
    }

    #[test]
    fn fixed_history_debug_invariants_hold() {
        let mut history = FixedHistory::<2>::new();
        history.record(1);
        history.record(2);
        history.record(3);
        history.debug_validate_invariants();
    }

    #[test]
    fn fixed_history_debug_snapshot_mru() {
        let mut history = FixedHistory::<3>::new();
        history.record(10);
        history.record(20);
        history.record(30);
        assert_eq!(history.debug_snapshot_mru(), vec![30, 20, 10]);
    }

    // -----------------------------------------------------------------------
    // iter() / IntoIterator tests
    // -----------------------------------------------------------------------

    #[test]
    fn iter_yields_mru_order() {
        let mut h = FixedHistory::<4>::new();
        h.record(10);
        h.record(20);
        h.record(30);

        let v: Vec<_> = h.iter().collect();
        assert_eq!(v, vec![30, 20, 10]);
    }

    #[test]
    fn iter_on_empty() {
        let h = FixedHistory::<4>::new();
        assert_eq!(h.iter().count(), 0);
    }

    #[test]
    fn iter_on_zero_capacity() {
        let h = FixedHistory::<0>::new();
        assert_eq!(h.iter().count(), 0);
    }

    #[test]
    fn iter_after_wrap() {
        let mut h = FixedHistory::<3>::new();
        for t in 1..=6 {
            h.record(t);
        }
        // Only last 3: 6, 5, 4
        let v: Vec<_> = h.iter().collect();
        assert_eq!(v, vec![6, 5, 4]);
    }

    #[test]
    fn iter_partially_filled() {
        let mut h = FixedHistory::<5>::new();
        h.record(100);
        h.record(200);

        let v: Vec<_> = h.iter().collect();
        assert_eq!(v, vec![200, 100]);
    }

    #[test]
    fn iter_count_matches_len() {
        let mut h = FixedHistory::<4>::new();
        for t in [10, 20, 30, 40, 50] {
            h.record(t);
            assert_eq!(h.iter().count(), h.len());
        }
    }

    #[test]
    fn iter_exact_size() {
        let mut h = FixedHistory::<3>::new();
        h.record(1);
        h.record(2);

        let mut it = h.iter();
        assert_eq!(it.len(), 2);
        it.next();
        assert_eq!(it.len(), 1);
        it.next();
        assert_eq!(it.len(), 0);
        assert!(it.next().is_none());
    }

    #[test]
    fn iter_matches_to_vec_mru() {
        let mut h = FixedHistory::<5>::new();
        for t in [10, 20, 30, 40, 50, 60] {
            h.record(t);
        }
        let from_iter: Vec<_> = h.iter().collect();
        assert_eq!(from_iter, h.to_vec_mru());
    }

    #[test]
    fn ref_into_iter_for_loop() {
        let mut h = FixedHistory::<3>::new();
        h.record(10);
        h.record(20);

        let mut sum = 0u64;
        for t in &h {
            sum += t;
        }
        assert_eq!(sum, 30);
        assert_eq!(h.len(), 2); // not consumed
    }

    #[test]
    fn owned_into_iter_for_loop() {
        let mut h = FixedHistory::<3>::new();
        h.record(10);
        h.record(20);
        h.record(30);

        let mut sum = 0u64;
        for t in h {
            sum += t;
        }
        assert_eq!(sum, 60);
    }

    #[test]
    fn into_iter_exact_size() {
        let mut h = FixedHistory::<4>::new();
        h.record(1);
        h.record(2);
        h.record(3);

        let mut it = h.into_iter();
        assert_eq!(it.len(), 3);
        it.next();
        assert_eq!(it.len(), 2);
    }

    #[test]
    fn into_iter_yields_mru_order() {
        let mut h = FixedHistory::<3>::new();
        h.record(7);
        h.record(8);
        h.record(9);

        let v: Vec<_> = h.into_iter().collect();
        assert_eq!(v, vec![9, 8, 7]);
    }

    #[test]
    fn iter_after_clear() {
        let mut h = FixedHistory::<3>::new();
        h.record(1);
        h.record(2);
        h.clear();

        assert_eq!(h.iter().count(), 0);
        assert_eq!(h.into_iter().count(), 0);
    }

    // -----------------------------------------------------------------------
    // PartialEq / Eq tests
    // -----------------------------------------------------------------------

    #[test]
    fn eq_same_entries_same_order() {
        let mut a = FixedHistory::<3>::new();
        let mut b = FixedHistory::<3>::new();
        a.record(10);
        a.record(20);
        a.record(30);
        b.record(10);
        b.record(20);
        b.record(30);

        assert_eq!(a, b);
    }

    #[test]
    fn eq_different_cursor_same_logical_content() {
        // Same logical timestamps but different cursor positions due to wrapping
        let mut a = FixedHistory::<3>::new();
        a.record(1);
        a.record(2);
        a.record(3);

        let mut b = FixedHistory::<3>::new();
        // Insert extra entries to advance cursor, but overwrite with same logical content
        b.record(99);
        b.record(1);
        b.record(2);
        b.record(3);

        assert_eq!(a, b);
    }

    #[test]
    fn ne_different_len() {
        let mut a = FixedHistory::<3>::new();
        a.record(10);
        let mut b = FixedHistory::<3>::new();
        b.record(10);
        b.record(20);

        assert_ne!(a, b);
    }

    #[test]
    fn ne_different_values() {
        let mut a = FixedHistory::<3>::new();
        a.record(10);
        a.record(20);
        let mut b = FixedHistory::<3>::new();
        b.record(10);
        b.record(99);

        assert_ne!(a, b);
    }

    #[test]
    fn eq_empty_histories() {
        let a = FixedHistory::<4>::new();
        let b = FixedHistory::<4>::new();
        assert_eq!(a, b);
    }

    #[test]
    fn eq_after_clear() {
        let mut a = FixedHistory::<3>::new();
        a.record(1);
        a.record(2);
        a.record(3);
        a.clear();

        let b = FixedHistory::<3>::new();
        assert_eq!(a, b);
    }

    // -----------------------------------------------------------------------
    // Hash tests
    // -----------------------------------------------------------------------

    #[test]
    fn hash_equal_histories_same_hash() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut a = FixedHistory::<3>::new();
        a.record(10);
        a.record(20);
        a.record(30);

        let mut b = FixedHistory::<3>::new();
        // Different cursor position, same logical content
        b.record(99);
        b.record(10);
        b.record(20);
        b.record(30);

        let hash_of = |h: &FixedHistory<3>| {
            let mut s = DefaultHasher::new();
            h.hash(&mut s);
            s.finish()
        };

        assert_eq!(hash_of(&a), hash_of(&b));
    }

    #[test]
    fn hash_usable_in_hashmap() {
        use std::collections::HashMap;

        let mut a = FixedHistory::<2>::new();
        a.record(1);
        a.record(2);

        let mut b = FixedHistory::<2>::new();
        b.record(1);
        b.record(2);

        let mut map = HashMap::new();
        map.insert(a, "entry");
        assert_eq!(map.get(&b), Some(&"entry"));
    }

    // -----------------------------------------------------------------------
    // Copy tests
    // -----------------------------------------------------------------------

    #[test]
    fn copy_produces_independent_value() {
        let mut original = FixedHistory::<3>::new();
        original.record(10);
        original.record(20);

        // Copy (implicit via binding)
        let copy = original;

        // Mutating original after copy doesn't affect copy
        original.record(99);
        assert_eq!(copy.most_recent(), Some(20));
        assert_eq!(original.most_recent(), Some(99));
    }

    #[test]
    fn copy_can_be_passed_by_value() {
        fn sum_history(h: FixedHistory<3>) -> u64 {
            h.iter().sum()
        }

        let mut h = FixedHistory::<3>::new();
        h.record(10);
        h.record(20);
        h.record(30);

        // Call twice — possible only because FixedHistory is Copy
        assert_eq!(sum_history(h), 60);
        assert_eq!(sum_history(h), 60);
    }

    // -----------------------------------------------------------------------
    // Security / hardening tests
    // -----------------------------------------------------------------------

    #[test]
    fn max_k_bound_permits_realistic_k() {
        let h = FixedHistory::<{ MAX_K }>::new();
        assert_eq!(h.capacity(), MAX_K);
        assert!(h.is_empty());
    }

    #[test]
    fn kth_most_recent_handles_full_capacity_at_max_k() {
        // Exercises the `cursor + (K - k)` path with the largest allowed K
        // to confirm no intermediate arithmetic overflow.
        let mut h = FixedHistory::<{ MAX_K }>::new();
        for ts in 0..(MAX_K as u64) {
            h.record(ts);
        }
        assert_eq!(h.most_recent(), Some((MAX_K as u64) - 1));
        assert_eq!(h.kth_most_recent(MAX_K), Some(0));
        assert_eq!(h.kth_most_recent(MAX_K + 1), None);
    }

    // -----------------------------------------------------------------------
    // Stale-slot hardening tests
    // -----------------------------------------------------------------------

    #[test]
    fn clear_zeroes_backing_array() {
        // After clear(), no sentinel previously-recorded timestamp should
        // remain in memory where `Debug` / core dumps could observe it.
        let mut h = FixedHistory::<4>::new();
        h.record(0xDEAD_BEEF);
        h.record(0xCAFE_BABE);
        h.record(0xFEED_FACE);

        h.clear();

        // Public API already hid them, but check the raw array directly.
        for slot in h.data.iter() {
            assert_eq!(*slot, 0, "clear() must zero the backing array");
        }
    }

    #[test]
    fn debug_output_hides_stale_slots_after_wrap() {
        // Record more than K entries so the ring wraps, then make sure
        // `Debug` output does not mention the overwritten timestamp.
        let mut h = FixedHistory::<2>::new();
        h.record(0x1111_1111_1111_1111);
        h.record(0x2222_2222_2222_2222);
        h.record(0x3333_3333_3333_3333); // overwrites 0x1111...

        let rendered = format!("{h:?}");
        assert!(
            !rendered.contains("1111111111111111"),
            "Debug must not expose overwritten timestamp: {rendered}"
        );
        // Live entries should still appear.
        assert!(rendered.contains("len: 2"), "debug output: {rendered}");
    }

    #[test]
    fn debug_output_hides_stale_slots_after_clear() {
        let mut h = FixedHistory::<3>::new();
        h.record(0xAAAA_AAAA_AAAA_AAAA);
        h.record(0xBBBB_BBBB_BBBB_BBBB);

        h.clear();

        let rendered = format!("{h:?}");
        assert!(
            !rendered.contains("AAAA") && !rendered.contains("aaaa"),
            "Debug must not expose cleared timestamps: {rendered}"
        );
        assert!(rendered.contains("len: 0"), "debug output: {rendered}");
    }

    #[test]
    fn boxed_returns_empty_history() {
        let h: Box<FixedHistory<8>> = FixedHistory::boxed();
        assert!(h.is_empty());
        assert_eq!(h.len(), 0);
        assert_eq!(h.capacity(), 8);
        assert_eq!(h.most_recent(), None);
        h.debug_validate_invariants();
    }

    #[test]
    fn boxed_is_usable_like_new() {
        let mut h: Box<FixedHistory<4>> = FixedHistory::boxed();
        h.record(10);
        h.record(20);
        h.record(30);
        assert_eq!(h.most_recent(), Some(30));
        assert_eq!(h.to_vec_mru(), vec![30, 20, 10]);
    }

    #[test]
    fn boxed_at_max_k_succeeds() {
        // The whole point of `boxed()`: allocate 32 KiB of ring buffer
        // without touching the stack.
        let mut h: Box<FixedHistory<{ MAX_K }>> = FixedHistory::boxed();
        h.record(42);
        assert_eq!(h.most_recent(), Some(42));
        assert_eq!(h.len(), 1);
    }

    #[test]
    fn boxed_at_zero_capacity_is_noop() {
        let mut h: Box<FixedHistory<0>> = FixedHistory::boxed();
        h.record(1);
        assert!(h.is_empty());
        assert_eq!(h.most_recent(), None);
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    // =============================================================================
    // Property Tests - Core Invariants
    // =============================================================================

    proptest! {
        /// Property: len() never exceeds capacity K
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_len_within_capacity(
            timestamps in prop::collection::vec(any::<u64>(), 0..100)
        ) {
            let mut history = FixedHistory::<10>::new();

            for ts in timestamps {
                history.record(ts);
                prop_assert!(history.len() <= history.capacity());
                prop_assert!(history.len() <= 10);
            }
        }

        /// Property: most_recent() returns the last recorded timestamp
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_most_recent_is_last_recorded(
            timestamps in prop::collection::vec(any::<u64>(), 1..50)
        ) {
            let mut history = FixedHistory::<8>::new();

            for &ts in &timestamps {
                history.record(ts);
                prop_assert_eq!(history.most_recent(), Some(ts));
            }
        }

        /// Property: to_vec_mru() length matches len()
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_vec_mru_length_matches_len(
            timestamps in prop::collection::vec(any::<u64>(), 0..50)
        ) {
            let mut history = FixedHistory::<7>::new();

            for ts in timestamps {
                history.record(ts);
                let vec = history.to_vec_mru();
                prop_assert_eq!(vec.len(), history.len());
            }
        }

        /// Property: kth_most_recent(k) matches to_vec_mru()[k-1]
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_kth_matches_vec_mru(
            timestamps in prop::collection::vec(any::<u64>(), 1..50)
        ) {
            let mut history = FixedHistory::<6>::new();

            for ts in timestamps {
                history.record(ts);
            }

            let vec = history.to_vec_mru();
            for k in 1..=history.len() {
                prop_assert_eq!(history.kth_most_recent(k), Some(vec[k - 1]));
            }
        }

        /// Property: Invariants hold after every operation
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_invariants_always_hold(
            timestamps in prop::collection::vec(any::<u64>(), 0..100)
        ) {
            let mut history = FixedHistory::<5>::new();

            for ts in timestamps {
                history.record(ts);
                history.debug_validate_invariants();
            }
        }
    }

    // =============================================================================
    // Property Tests - Boundary Conditions
    // =============================================================================

    proptest! {
        /// Property: k=0 always returns None
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_k_zero_returns_none(
            timestamps in prop::collection::vec(any::<u64>(), 0..30)
        ) {
            let mut history = FixedHistory::<5>::new();

            for ts in timestamps {
                history.record(ts);
                prop_assert_eq!(history.kth_most_recent(0), None);
            }
        }

        /// Property: k > len() returns None
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_k_exceeds_len_returns_none(
            timestamps in prop::collection::vec(any::<u64>(), 0..20),
            k in 1usize..100
        ) {
            let mut history = FixedHistory::<8>::new();

            for ts in timestamps {
                history.record(ts);
            }

            if k > history.len() {
                prop_assert_eq!(history.kth_most_recent(k), None);
            }
        }

        /// Property: Empty history returns None for all operations
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_empty_returns_none(k in 0usize..20) {
            let history = FixedHistory::<10>::new();

            prop_assert!(history.is_empty());
            prop_assert_eq!(history.len(), 0);
            prop_assert_eq!(history.most_recent(), None);
            prop_assert_eq!(history.kth_most_recent(k), None);
            prop_assert_eq!(history.to_vec_mru().len(), 0);
        }
    }

    // =============================================================================
    // Property Tests - Ring Buffer Wrapping
    // =============================================================================

    proptest! {
        /// Property: After K+N records, only last K timestamps are retained
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_wrapping_retains_last_k(
            timestamps in prop::collection::vec(1u64..1000, 10..50)
        ) {
            const K: usize = 7;
            let mut history = FixedHistory::<K>::new();

            for ts in &timestamps {
                history.record(*ts);
            }

            // History should contain at most K elements
            prop_assert_eq!(history.len(), K.min(timestamps.len()));

            // Verify it contains the last K timestamps
            let expected_len = K.min(timestamps.len());
            let expected: Vec<u64> = timestamps[timestamps.len() - expected_len..]
                .iter()
                .rev()
                .copied()
                .collect();

            prop_assert_eq!(history.to_vec_mru(), expected);
        }

        /// Property: Order is preserved in MRU order after wrapping
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_order_preserved_after_wrapping(
            timestamps in prop::collection::vec(any::<u64>(), 5..50)
        ) {
            const K: usize = 5;
            let mut history = FixedHistory::<K>::new();

            for ts in &timestamps {
                history.record(*ts);
            }

            let vec = history.to_vec_mru();

            // Verify strict MRU order: each element should equal kth_most_recent
            for (idx, &val) in vec.iter().enumerate() {
                prop_assert_eq!(history.kth_most_recent(idx + 1), Some(val));
            }
        }

        /// Property: Length increases monotonically until K, then stays at K
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_length_grows_then_saturates(
            timestamps in prop::collection::vec(any::<u64>(), 1..30)
        ) {
            const K: usize = 8;
            let mut history = FixedHistory::<K>::new();

            for (idx, ts) in timestamps.iter().enumerate() {
                history.record(*ts);
                let expected_len = (idx + 1).min(K);
                prop_assert_eq!(history.len(), expected_len);
            }
        }
    }

    // =============================================================================
    // Property Tests - Clear Operations
    // =============================================================================

    proptest! {
        /// Property: clear() resets to empty state
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_clear_resets_state(
            timestamps in prop::collection::vec(any::<u64>(), 1..30)
        ) {
            let mut history = FixedHistory::<6>::new();

            for ts in timestamps {
                history.record(ts);
            }

            history.clear();

            prop_assert!(history.is_empty());
            prop_assert_eq!(history.len(), 0);
            prop_assert_eq!(history.most_recent(), None);
            prop_assert_eq!(history.to_vec_mru().len(), 0);
            history.debug_validate_invariants();
        }

        /// Property: clear_shrink() behaves identically to clear()
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_clear_shrink_same_as_clear(
            timestamps in prop::collection::vec(any::<u64>(), 1..30)
        ) {
            let mut history1 = FixedHistory::<6>::new();
            let mut history2 = FixedHistory::<6>::new();

            for ts in &timestamps {
                history1.record(*ts);
                history2.record(*ts);
            }

            history1.clear();
            history2.clear_shrink();

            prop_assert_eq!(history1.len(), history2.len());
            prop_assert_eq!(history1.is_empty(), history2.is_empty());
            prop_assert_eq!(history1.capacity(), history2.capacity());
        }

        /// Property: Can record after clear
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_usable_after_clear(
            timestamps1 in prop::collection::vec(any::<u64>(), 1..20),
            timestamps2 in prop::collection::vec(any::<u64>(), 1..20)
        ) {
            let mut history = FixedHistory::<5>::new();

            for ts in timestamps1 {
                history.record(ts);
            }

            history.clear();

            for ts in &timestamps2 {
                history.record(*ts);
            }

            prop_assert_eq!(history.len(), timestamps2.len().min(5));
            prop_assert_eq!(history.most_recent(), Some(*timestamps2.last().unwrap()));
        }
    }

    // =============================================================================
    // Property Tests - Zero Capacity Edge Case
    // =============================================================================

    proptest! {
        /// Property: Zero capacity history is always empty
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_zero_capacity_always_empty(
            timestamps in prop::collection::vec(any::<u64>(), 0..30)
        ) {
            let mut history = FixedHistory::<0>::new();

            for ts in timestamps {
                history.record(ts);
                prop_assert!(history.is_empty());
                prop_assert_eq!(history.len(), 0);
                prop_assert_eq!(history.capacity(), 0);
                prop_assert_eq!(history.most_recent(), None);
            }
        }
    }

    // =============================================================================
    // Property Tests - Reference Implementation Equivalence
    // =============================================================================

    proptest! {
        /// Property: Behavior matches reference Vec implementation
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_matches_reference_implementation(
            timestamps in prop::collection::vec(any::<u64>(), 0..50)
        ) {
            const K: usize = 10;
            let mut history = FixedHistory::<K>::new();
            let mut reference: Vec<u64> = Vec::new();

            for ts in timestamps {
                history.record(ts);
                reference.push(ts);

                // Keep only last K in reference
                if reference.len() > K {
                    reference.remove(0);
                }

                // Verify length matches
                prop_assert_eq!(history.len(), reference.len());

                // Verify most_recent matches
                if !reference.is_empty() {
                    prop_assert_eq!(history.most_recent(), Some(*reference.last().unwrap()));
                }

                // Verify to_vec_mru matches (reference is in LRU order, so reverse it)
                let expected: Vec<u64> = reference.iter().rev().copied().collect();
                prop_assert_eq!(history.to_vec_mru(), expected);

                // Verify each kth_most_recent
                for k in 1..=reference.len() {
                    let expected_idx = reference.len() - k;
                    prop_assert_eq!(history.kth_most_recent(k), Some(reference[expected_idx]));
                }
            }
        }
    }
}
