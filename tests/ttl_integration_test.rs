//! Black-box semantics tests for the TTL layer.
//!
//! Driven by `MockClock` so timing is deterministic. The tests exercise
//! the [`DynExpiringCache`] handed out by
//! [`CacheBuilder::with_default_ttl`], plus the raw
//! [`Expiring<…>`](cachekit::policy::expiring::Expiring) decorator wrapped
//! around a couple of concrete policies.

#![cfg(feature = "ttl")]

use std::time::Duration;

use cachekit::builder::{CacheBuilder, CachePolicy};
use cachekit::policy::expiring::Expiring;
use cachekit::policy::fast_lru::FastLru;
use cachekit::time::MockClock;
use cachekit::traits::{Cache, ExpiringCache, TtlStatus};

/// Common setup: a deterministic `Expiring<FastLru<u64, String>>` with the
/// supplied default TTL.
fn make_expiring(
    capacity: usize,
    default_ttl: Option<Duration>,
) -> Expiring<FastLru<u64, String>, u64, String, MockClock> {
    let clock = MockClock::new();
    let inner = FastLru::<u64, String>::new(capacity);
    Expiring::with_default_ttl(inner, clock, default_ttl)
}

// ---------------------------------------------------------------------------
// Decorator semantics
// ---------------------------------------------------------------------------

#[test]
fn insert_with_zero_ttl_removes_existing_and_returns_none() {
    let mut cache = make_expiring(8, None);
    cache.insert_with_ttl(1, "a".into(), Duration::from_secs(60));

    // Zero TTL removes the entry; previous live value is returned.
    assert_eq!(
        cache.insert_with_ttl(1, "b".into(), Duration::ZERO),
        Some("a".to_string())
    );
    assert!(!cache.contains(&1));
}

#[test]
fn expired_entry_observed_as_missing_on_read() {
    let mut cache = make_expiring(8, None);
    cache.insert_with_ttl(1, "a".into(), Duration::from_millis(100));
    cache.clock().advance(Duration::from_millis(99));
    assert_eq!(cache.get(&1), Some(&"a".to_string()));

    cache.clock().advance(Duration::from_millis(1));
    assert_eq!(cache.get(&1), None);
    assert_eq!(cache.len(), 0);
}

#[test]
fn insert_over_expired_entry_returns_none() {
    let mut cache = make_expiring(8, None);
    cache.insert_with_ttl(1, "old".into(), Duration::from_millis(50));
    cache.clock().advance(Duration::from_millis(60));

    // Plain insert with no default TTL — the previous entry is expired,
    // so the return value is None rather than the stale "old".
    assert_eq!(cache.insert(1, "new".into()), None);
    assert_eq!(cache.peek(&1), Some(&"new".to_string()));
}

#[test]
fn set_ttl_extension_works_only_on_live_entries() {
    let mut cache = make_expiring(8, None);
    cache.insert_with_ttl(1, "a".into(), Duration::from_millis(50));
    cache.clock().advance(Duration::from_millis(40));

    // Extend live entry.
    assert!(cache.set_ttl(&1, Duration::from_millis(200)));
    cache.clock().advance(Duration::from_millis(100));
    assert!(cache.contains(&1));

    // Now let it expire and try to extend.
    cache.clock().advance(Duration::from_millis(200));
    assert!(!cache.set_ttl(&1, Duration::from_millis(50)));
    // Side effect: expired entry was purged.
    assert!(!cache.contains(&1));
}

#[test]
fn purge_expired_returns_exact_count() {
    let mut cache = make_expiring(16, None);
    for i in 0..5 {
        cache.insert_with_ttl(i, format!("{i}"), Duration::from_millis(10));
    }
    for i in 5..10 {
        cache.insert_with_ttl(i, format!("{i}"), Duration::from_secs(60));
    }
    cache.clock().advance(Duration::from_millis(50));
    assert_eq!(cache.purge_expired(), 5);
    assert_eq!(cache.len(), 5);
}

#[test]
fn per_entry_ttl_overrides_default_including_zero() {
    let mut cache = make_expiring(8, Some(Duration::from_secs(60)));

    // Per-entry TTL shorter than default.
    cache.insert_with_ttl(1, "short".into(), Duration::from_millis(10));
    cache.clock().advance(Duration::from_millis(20));
    assert!(!cache.contains(&1));

    // Per-entry zero overrides default.
    cache.insert_with_ttl(2, "instant".into(), Duration::ZERO);
    assert!(!cache.contains(&2));

    // Plain insert inherits the default.
    cache.insert(3, "inherits".into());
    cache.clock().advance(Duration::from_secs(59));
    assert!(cache.contains(&3));
    cache.clock().advance(Duration::from_secs(2));
    assert!(cache.get(&3).is_none());
}

#[test]
fn ttl_status_reports_all_four_states() {
    let mut cache = make_expiring(8, None);

    // Missing.
    assert_eq!(cache.ttl_status(&1), TtlStatus::Missing);

    // Immortal (no TTL set, no default).
    cache.insert(2, "forever".into());
    assert_eq!(cache.ttl_status(&2), TtlStatus::Immortal);

    // Live.
    cache.insert_with_ttl(3, "live".into(), Duration::from_millis(500));
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

    // Expired (resident but past deadline).
    cache.clock().advance(Duration::from_millis(600));
    assert_eq!(cache.ttl_status(&3), TtlStatus::Expired);
}

// ---------------------------------------------------------------------------
// Builder integration
// ---------------------------------------------------------------------------

#[test]
fn builder_with_default_ttl_produces_dyn_expiring_cache() {
    let mut cache = CacheBuilder::new(8)
        .with_default_ttl(Duration::from_secs(60))
        .build::<u64, String>(CachePolicy::FastLru);

    assert_eq!(cache.default_ttl(), Some(Duration::from_secs(60)));
    cache.insert(1, "value".into());
    assert_eq!(cache.peek(&1), Some(&"value".to_string()));
    cache.insert_with_ttl(2, "short".into(), Duration::from_millis(10));
    assert_eq!(cache.len(), 2);
}

#[test]
fn dyn_expiring_cache_with_lru_policy_round_trips() {
    let mut cache = CacheBuilder::new(8)
        .with_default_ttl(Duration::from_secs(60))
        .build::<u64, String>(CachePolicy::Lru);

    cache.insert(1, "a".into());
    cache.insert(2, "b".into());
    assert_eq!(cache.get(&1), Some(&"a".to_string()));
    assert_eq!(cache.live_len(), 2);
    cache.clear();
    assert_eq!(cache.len(), 0);
}

#[test]
fn dyn_expiring_cache_purge_expired_works_via_builder_path() {
    let mut cache = CacheBuilder::new(8)
        .with_default_ttl(Duration::from_millis(50))
        .build::<u64, String>(CachePolicy::FastLru);

    cache.insert(1, "x".into());
    cache.insert(2, "y".into());
    cache.insert_with_ttl(3, "long".into(), Duration::from_secs(60));

    // Without a way to advance StdClock, just confirm the API shape.
    // Real expiry-on-read is exercised via the MockClock-backed tests above.
    assert_eq!(cache.purge_expired(), 0);
    assert_eq!(cache.live_len(), 3);
}

// ---------------------------------------------------------------------------
// Property test: random op sequence against a reference model.
// ---------------------------------------------------------------------------

mod proptests {
    use std::collections::HashMap;
    use std::time::Duration;

    use cachekit::policy::expiring::Expiring;
    use cachekit::policy::fast_lru::FastLru;
    use cachekit::time::MockClock;
    use cachekit::traits::{Cache, ExpiringCache};
    use proptest::prelude::*;

    /// Operations the reference model and the cache will both execute.
    #[derive(Debug, Clone)]
    enum Op {
        Insert { key: u8, ttl_ms: u32 },
        Get { key: u8 },
        Advance { delta_ms: u32 },
        Purge,
    }

    /// Reference: key -> deadline (in ms). Keys with deadline <= now are
    /// considered expired even before a `Purge` op.
    #[derive(Default)]
    struct RefModel {
        deadlines: HashMap<u8, u64>,
        now: u64,
        capacity: usize,
    }

    impl RefModel {
        fn new(capacity: usize) -> Self {
            Self {
                capacity,
                ..Default::default()
            }
        }

        fn insert(&mut self, key: u8, ttl_ms: u32) {
            if ttl_ms == 0 {
                self.deadlines.remove(&key);
                return;
            }
            // Saturating arithmetic mirrors `Expiring::deadline_from`.
            let deadline = self.now.saturating_add(ttl_ms as u64).min(u64::MAX - 1);
            // Eviction model: FastLru with bounded capacity. Because we
            // do not model LRU order precisely, we permit any key to be
            // evicted; the assertion below is only on "is this key
            // *definitely live*?", not on every absent key being expired.
            self.deadlines.insert(key, deadline);
            if self.deadlines.len() > self.capacity {
                // Pick an arbitrary victim — we don't try to match the
                // policy exactly; we only insist on liveness for keys
                // that are present in BOTH the model and the cache.
                let victim = *self.deadlines.keys().next().unwrap();
                if victim != key {
                    self.deadlines.remove(&victim);
                }
            }
        }

        fn advance(&mut self, delta_ms: u32) {
            self.now = self.now.saturating_add(delta_ms as u64);
        }

        /// `Some(true)` if the key is definitely live in the model;
        /// `Some(false)` if it is definitely expired; `None` if its state
        /// is undetermined (e.g. evicted under the FastLru policy in a
        /// way the model doesn't track).
        fn is_definitely_live(&self, key: u8) -> Option<bool> {
            match self.deadlines.get(&key) {
                Some(&d) if d > self.now => Some(true),
                Some(_) => Some(false),
                None => None,
            }
        }
    }

    fn op_strategy() -> impl Strategy<Value = Op> {
        prop_oneof![
            (any::<u8>(), 1u32..2_000).prop_map(|(k, t)| Op::Insert { key: k, ttl_ms: t }),
            any::<u8>().prop_map(|k| Op::Get { key: k }),
            (1u32..500).prop_map(|d| Op::Advance { delta_ms: d }),
            Just(Op::Purge),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 256, ..ProptestConfig::default() })]

        #[cfg_attr(miri, ignore)]
        #[test]
        fn random_ops_match_reference_model(ops in prop::collection::vec(op_strategy(), 0..120)) {
            const CAPACITY: usize = 16;

            let clock = MockClock::new();
            let inner: FastLru<u8, ()> = FastLru::new(CAPACITY);
            let mut cache = Expiring::with_default_ttl(inner, clock, None);
            let mut model = RefModel::new(CAPACITY);

            for op in ops {
                match op {
                    Op::Insert { key, ttl_ms } => {
                        cache.insert_with_ttl(key, (), Duration::from_millis(ttl_ms as u64));
                        model.insert(key, ttl_ms);
                    }
                    Op::Get { key } => {
                        let hit = cache.get(&key).is_some();
                        if let Some(true) = model.is_definitely_live(key) {
                            prop_assert!(
                                hit,
                                "model says key {key} is live but cache miss"
                            );
                        }
                        if let Some(false) = model.is_definitely_live(key) {
                            prop_assert!(
                                !hit,
                                "model says key {key} is expired but cache hit"
                            );
                        }
                    }
                    Op::Advance { delta_ms } => {
                        cache.clock().advance(Duration::from_millis(delta_ms as u64));
                        model.advance(delta_ms);
                    }
                    Op::Purge => {
                        let _ = cache.purge_expired();
                        // Model side: drop everything <= now.
                        let now = model.now;
                        model.deadlines.retain(|_, d| *d > now);
                    }
                }
            }
        }
    }
}
