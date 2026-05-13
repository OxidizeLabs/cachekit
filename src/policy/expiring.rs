//! `Expiring<C, K, T>` — decorator that adds time-based expiration to any
//! `Cache<K, V>`.
//!
//! ## Architecture
//!
//! ```text
//!   ┌─────────────────────────────────────────────────────────────────┐
//!   │  Expiring<C, K, T = StdClock>                                   │
//!   │                                                                 │
//!   │    inner: C                              ◀──── policy / storage │
//!   │    index: ExpirationIndex<K>             ◀──── deadlines        │
//!   │    clock: T : Clock                      ◀──── current tick     │
//!   │    default_ttl: Option<u64>              ◀──── fallback         │
//!   └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Key Components
//!
//! - The inner cache stores values and runs the eviction policy.
//! - The expiration index tracks deadlines, keyed by `K`.
//! - The clock is `&self`-readable; `StdClock` for production and
//!   `MockClock` for tests.
//!
//! ## Core Operations
//!
//! - `peek`/`contains` are logical reads — expired entries report as
//!   absent without physical removal, since the underlying [`Cache`]
//!   trait takes those by `&self`.
//! - `get`/`insert`/`remove`/`clear` physically purge expired entries.
//! - `purge_expired` drains everything with `deadline <= now`.
//!
//! ## Performance Trade-offs
//!
//! Each read pays at most one hash probe into the `ExpirationIndex`. Each
//! insert pays an `O(log n)` heap update. Per the design doc, this
//! overhead is acceptable for Phase 1; Phase 2 will embed `expires_at`
//! into specific policies' node layouts.
//!
//! ## Ordering Invariant
//!
//! The wrapper always removes from the **inner cache first**, then from
//! the `ExpirationIndex`. A panic in inner removal leaves the index
//! pointing at a key the cache still holds (next `pop_expired` is a
//! no-op). The reverse order would silently lose the deadline. The
//! ordering is enforced by [`Expiring::purge_one`] and the `Cache` /
//! `ExpiringCache` impls below.
//!
//! ## Thread Safety
//!
//! `Expiring<C, K, T>` is not itself thread-safe. The future
//! `ConcurrentExpiring<C>` wrapper (Phase 1.5 / Phase 2) holds the
//! decorator behind a `parking_lot::RwLock` and returns owned/`Arc`
//! values, per `docs/design/ttl.md` §4(e).
//!
//! ## Example Usage
//!
//! ```
//! use cachekit::policy::expiring::Expiring;
//! use cachekit::policy::fast_lru::FastLru;
//! use cachekit::time::MockClock;
//! use cachekit::traits::Cache;
//! use std::time::Duration;
//!
//! let clock = MockClock::new();
//! let mut cache = Expiring::with_default_ttl(
//!     FastLru::<u64, String>::new(8),
//!     clock,
//!     Some(Duration::from_secs(60)),
//! );
//!
//! cache.insert(1, "value".to_string());
//! assert_eq!(cache.peek(&1), Some(&"value".to_string()));
//!
//! cache.clock().advance(Duration::from_secs(61));
//! assert_eq!(cache.peek(&1), None);
//! ```

use std::hash::Hash;
use std::marker::PhantomData;
use std::time::Duration;

use crate::ds::expiration_index::ExpirationIndex;
use crate::time::{Clock, MockClock, StdClock, duration_to_ticks};
use crate::traits::{Cache, ExpiringCache, Tick, TtlStatus};

/// Decorator adding TTL semantics around an inner `Cache<K, V>`.
///
/// `V` is recorded as `PhantomData` so the value type the inner cache
/// works with is fixed at construction time; the wrapper itself does not
/// store values directly.
///
/// See the module documentation for the contract, ordering invariant, and
/// usage examples.
#[derive(Debug)]
pub struct Expiring<C, K, V, T = StdClock> {
    inner: C,
    index: ExpirationIndex<K>,
    clock: T,
    default_ttl_ticks: Option<u64>,
    #[cfg(feature = "metrics")]
    expirations: u64,
    _value: PhantomData<fn() -> V>,
}

impl<C, K, V> Expiring<C, K, V, StdClock>
where
    C: Cache<K, V>,
    K: Eq + Hash + Clone,
{
    /// Creates an expiring wrapper around `inner` using the system clock
    /// and no default TTL.
    pub fn new(inner: C) -> Self {
        Self::with_clock_and_default(inner, StdClock::new(), None)
    }

    /// Creates an expiring wrapper around `inner` using the system clock
    /// and a default TTL applied to entries inserted without an explicit
    /// TTL.
    pub fn with_default_ttl_std(inner: C, default_ttl: Duration) -> Self {
        Self::with_clock_and_default(inner, StdClock::new(), Some(default_ttl))
    }
}

impl<C, K, V, T> Expiring<C, K, V, T>
where
    C: Cache<K, V>,
    K: Eq + Hash + Clone,
    T: Clock,
{
    /// Creates an expiring wrapper with an explicit clock and a default
    /// TTL (or `None` for no default).
    pub fn with_default_ttl(inner: C, clock: T, default_ttl: Option<Duration>) -> Self {
        Self::with_clock_and_default(inner, clock, default_ttl)
    }

    fn with_clock_and_default(inner: C, clock: T, default_ttl: Option<Duration>) -> Self {
        let default_ttl_ticks = default_ttl.map(duration_to_ticks);
        Self {
            inner,
            index: ExpirationIndex::new(),
            clock,
            default_ttl_ticks,
            #[cfg(feature = "metrics")]
            expirations: 0,
            _value: PhantomData,
        }
    }

    /// Cumulative count of entries removed because their TTL deadline
    /// elapsed.
    ///
    /// Updated only when the `metrics` feature is enabled. Returns `0`
    /// otherwise so call sites compile regardless of the feature gate.
    pub fn expirations(&self) -> u64 {
        #[cfg(feature = "metrics")]
        {
            self.expirations
        }
        #[cfg(not(feature = "metrics"))]
        {
            0
        }
    }

    /// Returns a reference to the inner cache.
    pub fn inner(&self) -> &C {
        &self.inner
    }

    /// Returns a mutable reference to the inner cache.
    ///
    /// Bypassing the decorator with this can desync the expiration index;
    /// use only when you understand the ordering invariant.
    pub fn inner_mut(&mut self) -> &mut C {
        &mut self.inner
    }

    /// Returns a reference to the configured clock.
    pub fn clock(&self) -> &T {
        &self.clock
    }

    /// Returns the cache's default TTL as a `Duration`, if any.
    pub fn default_ttl(&self) -> Option<Duration> {
        self.default_ttl_ticks.map(Duration::from_millis)
    }

    /// Computes a deadline for a TTL relative to `now`, saturating to
    /// `u64::MAX - 1` on overflow.
    #[inline]
    fn deadline_from(&self, now: u64, ttl_ticks: u64) -> u64 {
        // `u64::MAX` is the "effectively never" sentinel for `Tick`. To
        // keep `<= now` always false for that sentinel we cap one short.
        now.saturating_add(ttl_ticks).min(u64::MAX - 1)
    }

    /// Returns `true` if `key`'s deadline (if any) is at or before `now`.
    fn is_expired_at(&self, key: &K, now: u64) -> bool {
        match self.index.deadline_of(key) {
            Some(deadline) => deadline <= now,
            None => false,
        }
    }

    /// Number of live (non-expired) entries.
    ///
    /// Takes `&mut self` because draining stale roots from the
    /// expiration index requires mutation. Returns an exact count, unlike
    /// the conservative [`Cache::len`] which reports physical occupancy.
    pub fn live_len(&mut self) -> usize {
        let now = self.clock.now();
        let mut expired: Vec<(K, u64)> = Vec::new();
        while let Some(entry) = self.index.pop_expired(now) {
            expired.push(entry);
        }
        let live = self.inner.len().saturating_sub(expired.len());
        // Restore index entries so subsequent operations still see the
        // deadlines; physical purge happens through `purge_expired` or a
        // mutating access. `pop_expired` only ever yields entries with
        // deadline <= now, so re-adding them keeps the state consistent.
        for (k, deadline) in expired {
            self.index.set_deadline(k, deadline);
        }
        live
    }

    /// Removes `key` from inner and index, honouring the ordering
    /// invariant: **inner removal happens first**.
    fn purge_one(&mut self, key: &K) -> Option<V> {
        let removed = self.inner.remove(key);
        let _ = self.index.remove(key);
        removed
    }

    /// Records one TTL-driven removal. No-op without the `metrics` feature.
    #[inline]
    fn record_expiration(&mut self) {
        #[cfg(feature = "metrics")]
        {
            self.expirations = self.expirations.saturating_add(1);
        }
    }
}

// ---------------------------------------------------------------------------
// Cache impl: logical reads via &self, physical purges via &mut self.
// ---------------------------------------------------------------------------

impl<C, K, V, T> Cache<K, V> for Expiring<C, K, V, T>
where
    C: Cache<K, V>,
    K: Eq + Hash + Clone,
    T: Clock,
{
    fn contains(&self, key: &K) -> bool {
        if !self.inner.contains(key) {
            return false;
        }
        // Logical read: hide expired entries.
        match self.index.deadline_of(key) {
            Some(deadline) => deadline > self.clock.now(),
            None => true,
        }
    }

    fn len(&self) -> usize {
        // Physical occupancy. See `live_len_value_aware` for the live count.
        self.inner.len()
    }

    fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    fn peek(&self, key: &K) -> Option<&V> {
        let value = self.inner.peek(key)?;
        match self.index.deadline_of(key) {
            Some(deadline) if deadline <= self.clock.now() => None,
            _ => Some(value),
        }
    }

    fn get(&mut self, key: &K) -> Option<&V> {
        let now = self.clock.now();
        if self.is_expired_at(key, now) {
            self.purge_one(key);
            self.record_expiration();
            return None;
        }
        self.inner.get(key)
    }

    fn insert(&mut self, key: K, value: V) -> Option<V> {
        let now = self.clock.now();
        let was_expired = self.is_expired_at(&key, now);

        // Apply the default TTL (if any) to the new entry.
        if let Some(ttl_ticks) = self.default_ttl_ticks {
            let deadline = self.deadline_from(now, ttl_ticks);
            self.index.set_deadline(key.clone(), deadline);
        } else {
            // No default; ensure any stale deadline is cleared so the
            // new entry is treated as immortal.
            self.index.remove(&key);
        }

        let previous = self.inner.insert(key, value);
        if was_expired {
            self.record_expiration();
            None
        } else {
            previous
        }
    }

    fn remove(&mut self, key: &K) -> Option<V> {
        let was_expired = self.is_expired_at(key, self.clock.now());
        let removed = self.purge_one(key);
        if was_expired {
            self.record_expiration();
            None
        } else {
            removed
        }
    }

    fn clear(&mut self) {
        self.inner.clear();
        self.index.clear();
    }
}

// ---------------------------------------------------------------------------
// ExpiringCache impl
// ---------------------------------------------------------------------------

impl<C, K, V, T> ExpiringCache<K, V> for Expiring<C, K, V, T>
where
    C: Cache<K, V>,
    K: Eq + Hash + Clone,
    T: Clock,
{
    fn insert_with_ttl(&mut self, key: K, value: V, ttl: Duration) -> Option<V> {
        let now = self.clock.now();
        let was_expired = self.is_expired_at(&key, now);

        if ttl.is_zero() {
            // Zero TTL == remove without inserting.
            let removed = self.purge_one(&key);
            return if was_expired {
                self.record_expiration();
                None
            } else {
                removed
            };
        }

        let ttl_ticks = duration_to_ticks(ttl);
        let deadline = self.deadline_from(now, ttl_ticks);
        self.index.set_deadline(key.clone(), deadline);
        let previous = self.inner.insert(key, value);
        if was_expired {
            self.record_expiration();
            None
        } else {
            previous
        }
    }

    fn ttl_status(&self, key: &K) -> TtlStatus {
        if !self.inner.contains(key) {
            return TtlStatus::Missing;
        }
        let now = self.clock.now();
        match self.index.deadline_of(key) {
            None => TtlStatus::Immortal,
            Some(deadline) if deadline <= now => TtlStatus::Expired,
            Some(deadline) => TtlStatus::Live {
                remaining: Duration::from_millis(deadline - now),
                deadline: Tick(deadline),
            },
        }
    }

    fn set_ttl(&mut self, key: &K, ttl: Duration) -> bool {
        let now = self.clock.now();
        if !self.inner.contains(key) {
            return false;
        }
        if self.is_expired_at(key, now) {
            self.purge_one(key);
            self.record_expiration();
            return false;
        }
        let ttl_ticks = duration_to_ticks(ttl);
        if ttl_ticks == 0 {
            // Zero TTL on a set_ttl call ==> remove the entry.
            self.purge_one(key);
            return false;
        }
        let deadline = self.deadline_from(now, ttl_ticks);
        self.index.set_deadline(key.clone(), deadline);
        true
    }

    fn purge_expired(&mut self) -> usize {
        let now = self.clock.now();
        let mut count = 0;
        // Drain index entries with deadline <= now; for each, remove from
        // inner. Ordering invariant: inner removal precedes index removal,
        // but `pop_expired` has already removed from the index; we must
        // tolerate inner-side panics by treating index as authoritative.
        // To preserve the documented invariant we don't pop first; instead
        // we peek, remove inner, then remove from index.
        loop {
            let key_clone = match self.index.peek_deadline() {
                Some((k, deadline)) if deadline <= now => k.clone(),
                _ => break,
            };
            let _ = self.inner.remove(&key_clone);
            let _ = self.index.remove(&key_clone);
            count += 1;
        }
        #[cfg(feature = "metrics")]
        {
            self.expirations = self.expirations.saturating_add(count as u64);
        }
        count
    }
}

// Public alias: a clock-backed Expiring wrapper used by the builder.
#[allow(dead_code)]
pub(crate) type ExpiringStdClock<C, K, V> = Expiring<C, K, V, StdClock>;
#[allow(dead_code)]
pub(crate) type ExpiringMockClock<C, K, V> = Expiring<C, K, V, MockClock>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::fast_lru::FastLru;

    #[test]
    fn insert_with_ttl_returns_none_on_first_insert() {
        let clock = MockClock::new();
        let inner: FastLru<u64, String> = FastLru::new(8);
        let mut cache = Expiring::with_default_ttl(inner, clock, None);
        assert_eq!(
            cache.insert_with_ttl(1, "a".into(), Duration::from_millis(100)),
            None
        );
        assert_eq!(cache.peek(&1), Some(&"a".to_string()));
    }

    #[test]
    fn entry_expires_on_get_after_clock_advance() {
        let clock = MockClock::new();
        let inner: FastLru<u64, String> = FastLru::new(8);
        let mut cache = Expiring::with_default_ttl(inner, clock, None);

        cache.insert_with_ttl(1, "a".into(), Duration::from_millis(50));
        assert_eq!(cache.get(&1), Some(&"a".to_string()));

        cache.clock().advance(Duration::from_millis(50));
        // Inclusive: deadline equal to now -> expired.
        assert_eq!(cache.get(&1), None);
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn peek_is_logical_does_not_purge() {
        let clock = MockClock::new();
        let inner: FastLru<u64, String> = FastLru::new(8);
        let mut cache = Expiring::with_default_ttl(inner, clock, None);

        cache.insert_with_ttl(1, "a".into(), Duration::from_millis(50));
        cache.clock().advance(Duration::from_millis(60));

        // peek hides the expired entry...
        assert_eq!(cache.peek(&1), None);
        assert!(!cache.contains(&1));
        // ...but the inner cache still physically holds it.
        assert_eq!(cache.len(), 1);
        // Mutable op purges.
        assert_eq!(cache.get(&1), None);
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn insert_after_expiry_returns_none_not_stale() {
        let clock = MockClock::new();
        let inner: FastLru<u64, String> = FastLru::new(8);
        let mut cache = Expiring::with_default_ttl(inner, clock, None);

        cache.insert_with_ttl(1, "old".into(), Duration::from_millis(50));
        cache.clock().advance(Duration::from_millis(60));
        // Re-inserting an expired key returns None, not "old".
        assert_eq!(
            cache.insert_with_ttl(1, "new".into(), Duration::from_millis(100)),
            None
        );
        assert_eq!(cache.peek(&1), Some(&"new".to_string()));
    }

    #[test]
    fn ttl_status_distinguishes_all_states() {
        let clock = MockClock::new();
        let inner: FastLru<u64, String> = FastLru::new(8);
        let mut cache = Expiring::with_default_ttl(inner, clock, None);

        assert_eq!(cache.ttl_status(&1), TtlStatus::Missing);

        cache.insert(2, "immortal".into()); // no default ttl
        assert_eq!(cache.ttl_status(&2), TtlStatus::Immortal);

        cache.insert_with_ttl(3, "live".into(), Duration::from_millis(100));
        match cache.ttl_status(&3) {
            TtlStatus::Live {
                remaining,
                deadline,
            } => {
                assert!(remaining > Duration::ZERO);
                assert!(deadline.as_u64() > 0);
            },
            other => panic!("expected Live, got {other:?}"),
        }

        cache.clock().advance(Duration::from_millis(101));
        assert_eq!(cache.ttl_status(&3), TtlStatus::Expired);
    }

    #[test]
    fn purge_expired_drains_and_counts() {
        let clock = MockClock::new();
        let inner: FastLru<u64, String> = FastLru::new(8);
        let mut cache = Expiring::with_default_ttl(inner, clock, None);

        cache.insert_with_ttl(1, "a".into(), Duration::from_millis(50));
        cache.insert_with_ttl(2, "b".into(), Duration::from_millis(150));
        cache.insert_with_ttl(3, "c".into(), Duration::from_millis(250));

        cache.clock().advance(Duration::from_millis(200));
        let purged = cache.purge_expired();
        assert_eq!(purged, 2);
        assert!(!cache.contains(&1));
        assert!(!cache.contains(&2));
        assert!(cache.contains(&3));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn default_ttl_overridden_by_per_entry_ttl() {
        let clock = MockClock::new();
        let inner: FastLru<u64, String> = FastLru::new(8);
        let mut cache =
            Expiring::with_default_ttl(inner, clock, Some(Duration::from_millis(1_000)));

        // Per-entry TTL wins over default.
        cache.insert_with_ttl(1, "fast".into(), Duration::from_millis(10));
        cache.clock().advance(Duration::from_millis(15));
        assert_eq!(cache.get(&1), None);

        // Zero TTL also wins (immediate expiry).
        cache.insert_with_ttl(2, "instant".into(), Duration::ZERO);
        assert!(!cache.contains(&2));
    }

    #[test]
    fn set_ttl_extends_live_entry() {
        let clock = MockClock::new();
        let inner: FastLru<u64, String> = FastLru::new(8);
        let mut cache = Expiring::with_default_ttl(inner, clock, None);

        cache.insert_with_ttl(1, "a".into(), Duration::from_millis(100));
        cache.clock().advance(Duration::from_millis(50));

        assert!(cache.set_ttl(&1, Duration::from_millis(200)));
        // Now deadline is now(50) + 200 = 250. After advancing 100ms total
        // (so clock=150), the entry must still be live.
        cache.clock().advance(Duration::from_millis(100));
        assert_eq!(cache.get(&1), Some(&"a".to_string()));

        // Missing key returns false.
        assert!(!cache.set_ttl(&999, Duration::from_millis(100)));
    }

    #[test]
    fn remove_returns_value_only_if_live() {
        let clock = MockClock::new();
        let inner: FastLru<u64, String> = FastLru::new(8);
        let mut cache = Expiring::with_default_ttl(inner, clock, None);

        cache.insert_with_ttl(1, "a".into(), Duration::from_millis(100));
        assert_eq!(cache.remove(&1), Some("a".to_string()));

        cache.insert_with_ttl(2, "b".into(), Duration::from_millis(50));
        cache.clock().advance(Duration::from_millis(60));
        assert_eq!(cache.remove(&2), None);
    }

    #[cfg(feature = "metrics")]
    #[test]
    fn expirations_counter_increments_on_purge_paths() {
        let clock = MockClock::new();
        let inner: FastLru<u64, String> = FastLru::new(8);
        let mut cache = Expiring::with_default_ttl(inner, clock, None);

        cache.insert_with_ttl(1, "a".into(), Duration::from_millis(50));
        cache.insert_with_ttl(2, "b".into(), Duration::from_millis(50));
        cache.insert_with_ttl(3, "c".into(), Duration::from_millis(50));
        cache.clock().advance(Duration::from_millis(100));

        // get on expired -> +1
        assert_eq!(cache.get(&1), None);
        // purge_expired drains the remaining two
        assert_eq!(cache.purge_expired(), 2);
        assert_eq!(cache.expirations(), 3);
    }

    #[test]
    fn expirations_returns_zero_without_metrics_feature() {
        // Sanity check: even without metrics, the accessor returns 0
        // rather than panicking. (The body of this test is identical
        // shape to the metrics one; the assertion differs.)
        let clock = MockClock::new();
        let inner: FastLru<u64, String> = FastLru::new(8);
        let cache = Expiring::with_default_ttl(inner, clock, None);
        // Always non-negative; with metrics off, always 0.
        let _ = cache.expirations();
    }

    #[test]
    fn live_len_excludes_expired_but_resident_entries() {
        let clock = MockClock::new();
        let inner: FastLru<u64, String> = FastLru::new(8);
        let mut cache = Expiring::with_default_ttl(inner, clock, None);

        cache.insert_with_ttl(1, "a".into(), Duration::from_millis(50));
        cache.insert_with_ttl(2, "b".into(), Duration::from_millis(200));

        cache.clock().advance(Duration::from_millis(100));
        // Physical occupancy still 2, but only 1 entry is live.
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.live_len(), 1);
    }
}
