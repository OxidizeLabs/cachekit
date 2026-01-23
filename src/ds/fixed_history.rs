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
//!
//! ## Operations
//!
//! | Operation         | Description                      | Complexity |
//! |-------------------|----------------------------------|------------|
//! | `record`          | Add timestamp (overwrites oldest)| O(1)       |
//! | `most_recent`     | Get most recent timestamp        | O(1)       |
//! | `kth_most_recent` | Get k-th most recent timestamp   | O(1)       |
//! | `to_vec_mru`      | Get all in MRU order             | O(K)       |
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
//! ## Implementation Notes
//!
//! - Uses a fixed-size array (no heap allocation)
//! - Const generic `K` determines history depth at compile time
//! - Zero-size history (`K=0`) is a no-op
//! - `debug_validate_invariants()` available in debug/test builds

/// Fixed-size ring buffer of the last `K` timestamps.
///
/// Stores timestamps in a circular buffer, automatically overwriting the oldest
/// entry when full. Provides O(1) access to any of the last K timestamps.
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
#[derive(Debug, Clone)]
pub struct FixedHistory<const K: usize> {
    data: [u64; K],
    len: usize,
    cursor: usize,
}

impl<const K: usize> FixedHistory<K> {
    /// Creates an empty history.
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
        Self {
            data: [0; K],
            len: 0,
            cursor: 0,
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
        let idx = (self.cursor + K - k) % K;
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

    /// Clears the history and resets cursor/length.
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

    #[cfg(any(test, debug_assertions))]
    /// Returns a debug snapshot of the history in MRU order.
    pub fn debug_snapshot_mru(&self) -> Vec<u64> {
        self.to_vec_mru()
    }

    #[cfg(any(test, debug_assertions))]
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
