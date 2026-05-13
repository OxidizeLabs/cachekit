//! Key-keyed deadline index used by the TTL decorator.
//!
//! ## Architecture
//!
//! ```text
//!   ┌──────────────────────────────────────────────────────────────┐
//!   │  ExpirationIndex<K>                                          │
//!   │                                                              │
//!   │    LazyMinHeap<K, u64>  (key -> deadline_ms)                 │
//!   │      ▲                                                       │
//!   │      ├── set_deadline(key, expires_at)                       │
//!   │      ├── remove(key)                                         │
//!   │      ├── peek_deadline()  -> earliest live (key, deadline)   │
//!   │      └── pop_expired(now) -> earliest live if deadline <= now │
//!   └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Key Components
//!
//! [`ExpirationIndex`] is a thin wrapper around
//! [`LazyMinHeap`](crate::ds::LazyMinHeap)`<K, u64>` with `auto_rebuild`
//! enabled. The wrapper hides the score type (always a `u64` deadline in
//! the cache's tick unit) and exposes operations specialised for TTL:
//!
//! - [`set_deadline`](ExpirationIndex::set_deadline) updates an entry's
//!   deadline, returning the previous deadline if any.
//! - [`peek_deadline`](ExpirationIndex::peek_deadline) returns references
//!   to the live entry with the earliest deadline.
//! - [`pop_expired`](ExpirationIndex::pop_expired) atomically peeks the
//!   earliest entry and, if its deadline `<= now`, removes and returns it.
//!
//! ## Performance Trade-offs
//!
//! - Insertion is `O(log n)` plus one key clone (the heap and the
//!   authoritative map both retain a copy).
//! - `peek_deadline` discards stale heap roots in place, so it is
//!   amortised `O(1)` between updates.
//! - Auto-rebuild defaults to factor `2`: stale heap entries are bounded
//!   at `2 * len()`. Callers that mutate every entry many times per
//!   epoch can tighten this with [`set_auto_rebuild`].
//!
//! ## Thread Safety
//!
//! `ExpirationIndex` is **not** thread-safe by itself; the TTL decorator
//! provides exterior locking. The wrapper inherits the underlying heap's
//! redacted `Debug` output, so no keys leak through tracing.
//!
//! ## Example Usage
//!
//! ```
//! use cachekit::ds::expiration_index::ExpirationIndex;
//!
//! let mut idx: ExpirationIndex<&str> = ExpirationIndex::new();
//! idx.set_deadline("a", 100);
//! idx.set_deadline("b", 50);
//!
//! assert_eq!(idx.peek_deadline(), Some((&"b", 50)));
//! assert_eq!(idx.pop_expired(40), None);          // none yet expired
//! assert_eq!(idx.pop_expired(60), Some(("b", 50))); // "b" expired at <=60
//! assert_eq!(idx.peek_deadline(), Some((&"a", 100)));
//! ```
//!
//! [`set_auto_rebuild`]: ExpirationIndex::set_auto_rebuild

use std::borrow::Borrow;
use std::hash::Hash;

use crate::ds::lazy_heap::LazyMinHeap;

/// Default rebuild factor: bound stale heap growth to `2 * live_len`.
///
/// Picked to keep amortised maintenance cheap while preventing heap bloat
/// from repeated `set_deadline` updates on the same key.
const DEFAULT_REBUILD_FACTOR: usize = 2;

/// Min-priority deadline index keyed by `K`.
///
/// Wraps a [`LazyMinHeap`]`<K, u64>` with auto-rebuild enabled. Deadlines
/// are opaque `u64` ticks; the unit is determined by the
/// [`Clock`](crate::time::Clock) the TTL decorator uses (conventionally
/// milliseconds in `cachekit`).
#[derive(Debug)]
pub struct ExpirationIndex<K> {
    heap: LazyMinHeap<K, u64>,
}

impl<K> ExpirationIndex<K>
where
    K: Eq + Hash + Clone,
{
    /// Creates an empty index with the default rebuild factor.
    pub fn new() -> Self {
        Self {
            heap: LazyMinHeap::with_auto_rebuild(DEFAULT_REBUILD_FACTOR),
        }
    }

    /// Creates an empty index with capacity pre-reserved for `capacity`
    /// distinct keys.
    ///
    /// Useful when the TTL decorator wraps a fixed-capacity cache.
    pub fn with_capacity(capacity: usize) -> Self {
        let mut heap = LazyMinHeap::with_capacity(capacity);
        heap.set_auto_rebuild(Some(DEFAULT_REBUILD_FACTOR));
        Self { heap }
    }

    /// Returns the number of live entries.
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Returns `true` if there are no live entries.
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Removes all entries.
    pub fn clear(&mut self) {
        self.heap.clear();
    }

    /// Sets `key`'s deadline and returns the previous deadline, if any.
    ///
    /// `expires_at` is in the cache's tick unit (typically milliseconds).
    /// A previous deadline is replaced; no validation against the current
    /// clock happens here.
    pub fn set_deadline(&mut self, key: K, expires_at: u64) -> Option<u64> {
        self.heap.update(key, expires_at)
    }

    /// Returns the current deadline for `key`, if any.
    pub fn deadline_of<Q>(&self, key: &Q) -> Option<u64>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.heap.score_of(key).copied()
    }

    /// Returns `true` if `key` has a deadline tracked here.
    pub fn contains<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.heap.score_of(key).is_some()
    }

    /// Removes `key` and returns its deadline, if any.
    pub fn remove<Q>(&mut self, key: &Q) -> Option<u64>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.heap.remove(key)
    }

    /// Returns the live entry with the earliest deadline without removing it.
    ///
    /// Discards stale heap roots in place; takes `&mut self` for that
    /// reason.
    pub fn peek_deadline(&mut self) -> Option<(&K, u64)> {
        self.heap.peek_best().map(|(k, s)| (k, *s))
    }

    /// Removes and returns the earliest entry if its deadline is `<= now`.
    ///
    /// The comparison is `<=` (not `<`): a deadline equal to `now` is
    /// already past in the chosen tick unit, matching the algorithm
    /// described in `docs/design/ttl.md` §3.
    pub fn pop_expired(&mut self, now: u64) -> Option<(K, u64)> {
        match self.peek_deadline() {
            Some((_, deadline)) if deadline <= now => self.heap.pop_best(),
            _ => None,
        }
    }

    /// Overrides the underlying heap's auto-rebuild factor.
    ///
    /// `None` disables auto-rebuild. Values below `1` are clamped to `1`.
    pub fn set_auto_rebuild(&mut self, factor: Option<usize>) {
        self.heap.set_auto_rebuild(factor);
    }
}

impl<K> Default for ExpirationIndex<K>
where
    K: Eq + Hash + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_deadline_returns_previous_and_updates() {
        let mut idx: ExpirationIndex<&str> = ExpirationIndex::new();
        assert_eq!(idx.set_deadline("a", 100), None);
        assert_eq!(idx.set_deadline("a", 200), Some(100));
        assert_eq!(idx.deadline_of(&"a"), Some(200));
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn peek_deadline_returns_earliest_live() {
        let mut idx: ExpirationIndex<&str> = ExpirationIndex::new();
        idx.set_deadline("a", 100);
        idx.set_deadline("b", 50);
        idx.set_deadline("c", 75);
        assert_eq!(idx.peek_deadline(), Some((&"b", 50)));
    }

    #[test]
    fn peek_deadline_skips_replaced_entries() {
        let mut idx: ExpirationIndex<&str> = ExpirationIndex::new();
        idx.set_deadline("a", 10);
        idx.set_deadline("a", 100); // earlier entry is stale
        idx.set_deadline("b", 50);
        assert_eq!(idx.peek_deadline(), Some((&"b", 50)));
    }

    #[test]
    fn pop_expired_respects_inclusive_comparison() {
        let mut idx: ExpirationIndex<&str> = ExpirationIndex::new();
        idx.set_deadline("a", 100);
        // 99 -> not yet expired
        assert_eq!(idx.pop_expired(99), None);
        // 100 -> expired (inclusive)
        assert_eq!(idx.pop_expired(100), Some(("a", 100)));
        assert_eq!(idx.pop_expired(200), None);
    }

    #[test]
    fn pop_expired_drains_in_deadline_order() {
        let mut idx: ExpirationIndex<&str> = ExpirationIndex::new();
        idx.set_deadline("a", 100);
        idx.set_deadline("b", 200);
        idx.set_deadline("c", 300);
        let mut out = Vec::new();
        while let Some(entry) = idx.pop_expired(250) {
            out.push(entry);
        }
        assert_eq!(out, vec![("a", 100), ("b", 200)]);
        // "c" not yet expired.
        assert_eq!(idx.deadline_of(&"c"), Some(300));
    }

    #[test]
    fn remove_clears_deadline() {
        let mut idx: ExpirationIndex<&str> = ExpirationIndex::new();
        idx.set_deadline("a", 100);
        assert_eq!(idx.remove(&"a"), Some(100));
        assert_eq!(idx.remove(&"a"), None);
        assert!(idx.is_empty());
    }

    #[test]
    fn auto_rebuild_bounds_stale_entries() {
        let mut idx: ExpirationIndex<u32> = ExpirationIndex::new();
        for i in 0..100 {
            idx.set_deadline(1, i as u64);
        }
        // With factor 2, heap_len should stay bounded relative to len=1.
        assert_eq!(idx.len(), 1);
        // Drain to confirm correctness despite stale churn.
        assert_eq!(idx.pop_expired(99), Some((1, 99)));
        assert!(idx.is_empty());
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// `peek_deadline` always returns the earliest live deadline.
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_peek_returns_earliest(
            entries in prop::collection::vec((any::<u16>(), any::<u32>()), 0..32)
        ) {
            let mut idx: ExpirationIndex<u16> = ExpirationIndex::new();
            // Build the index. Later writes for the same key replace earlier ones.
            use std::collections::HashMap;
            let mut latest: HashMap<u16, u64> = HashMap::new();
            for (k, s) in entries {
                let s = s as u64;
                idx.set_deadline(k, s);
                latest.insert(k, s);
            }
            let expected_min = latest.values().min().copied();
            let actual_min = idx.peek_deadline().map(|(_, d)| d);
            prop_assert_eq!(actual_min, expected_min);
        }

        /// `pop_expired` drains everything with deadline <= now in ascending order.
        #[cfg_attr(miri, ignore)]
        #[test]
        fn prop_pop_expired_drains_in_order(
            entries in prop::collection::vec((any::<u16>(), 0u64..1_000_000), 0..32),
            now in 0u64..1_000_000
        ) {
            let mut idx: ExpirationIndex<u16> = ExpirationIndex::new();
            use std::collections::HashMap;
            let mut latest: HashMap<u16, u64> = HashMap::new();
            for (k, s) in entries {
                idx.set_deadline(k, s);
                latest.insert(k, s);
            }

            let mut popped: Vec<(u16, u64)> = Vec::new();
            while let Some(e) = idx.pop_expired(now) {
                popped.push(e);
            }

            // 1. Order is non-decreasing in deadline.
            for win in popped.windows(2) {
                prop_assert!(win[0].1 <= win[1].1);
            }
            // 2. Every popped (k, d) was the latest live deadline for k and d <= now.
            for (k, d) in &popped {
                prop_assert_eq!(latest.get(k).copied(), Some(*d));
                prop_assert!(*d <= now);
            }
            // 3. Everything still in the index has deadline > now.
            for (k, d) in latest.iter() {
                if popped.iter().any(|(pk, _)| pk == k) {
                    continue;
                }
                prop_assert!(*d > now);
                prop_assert_eq!(idx.deadline_of(k), Some(*d));
            }
        }
    }
}
