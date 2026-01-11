//! Fixed-size access history ring.
//!
//! Stores the last `K` timestamps in a ring buffer, providing O(1) record
//! and O(1) access to the k-th most recent entry.
//!
//! ## Architecture
//!
//! ```text
//!   data: [u64; K]     cursor: write index
//!   len: number of valid entries (<= K)
//!
//!   data  = [10, 20, 30]
//!   cursor = 0   (next write overwrites oldest)
//!
//!   most_recent (k=1) => data[(cursor + K - 1) % K]
//!   kth_most_recent(k) => data[(cursor + K - k) % K]
//! ```
//!
//! ## Behavior
//! - `record(ts)`: writes at cursor, overwriting oldest when full
//! - `most_recent` / `kth_most_recent`: read from ring
//! - `to_vec_mru`: returns values from most-recent → least-recent
//!
//! ## Performance
//! - `record`: O(1)
//! - `kth_most_recent`: O(1)
//! - `to_vec_mru`: O(K)
//!
//! `debug_validate_invariants()` is available in debug/test builds.
#[derive(Debug, Clone)]
pub struct FixedHistory<const K: usize> {
    data: [u64; K],
    len: usize,
    cursor: usize,
}

impl<const K: usize> FixedHistory<K> {
    pub fn new() -> Self {
        Self {
            data: [0; K],
            len: 0,
            cursor: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        K
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

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

    pub fn most_recent(&self) -> Option<u64> {
        self.kth_most_recent(1)
    }

    pub fn kth_most_recent(&self, k: usize) -> Option<u64> {
        if K == 0 || k == 0 || k > self.len {
            return None;
        }
        let idx = (self.cursor + K - k) % K;
        Some(self.data[idx])
    }

    pub fn to_vec_mru(&self) -> Vec<u64> {
        (1..=self.len)
            .filter_map(|k| self.kth_most_recent(k))
            .collect()
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
}
