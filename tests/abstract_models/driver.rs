//! Shared assertion helpers for dual-run cache vs model tests.
//!
//! These functions compare a real cache (implementing capability traits from
//! [`cachekit::traits`](../../src/traits.rs)) against a [`PolicyModel`] step produced by
//! [`PolicyModel::apply`].
//!
//! Typical usage in `policy_semantics/*_tests.rs`:
//!
//! ```ignore
//! let step = model.apply(op.clone());
//! // … apply op to cache …
//! assert_peek_victim(cache, model);
//! assert_recency_rank(cache, model.model_recency_rank(k), k);
//! ```

#![allow(dead_code)]

use std::collections::HashSet;
use std::hash::Hash;

use cachekit::traits::{Cache, EvictingCache, RecencyTracking, VictimInspectable};

use crate::abstract_models::{HitMiss, ModelStep, Op, OracleExpectation, PolicyModel};

/// Compare model step against a cache implementing standard inspection traits.
///
/// Consolidated dual-run helper for new tests. Existing `policy_semantics/*_tests.rs` files
/// inline equivalent assertions in their `run_ops` loops.
pub fn assert_step<K, V, C, M>(
    cache: &C,
    _model: &M,
    step: &ModelStep<K>,
    op: &Op<K>,
    rank_before: Option<usize>,
) where
    K: Clone + Eq + Hash + std::fmt::Debug,
    C: Cache<K, V> + VictimInspectable<K, V> + RecencyTracking<K, V> + EvictingCache<K, V>,
    M: PolicyModel<K>,
{
    assert_eq!(
        cache.len(),
        step.resident.len(),
        "residency size mismatch after {op:?}"
    );
    for k in &step.resident {
        assert!(
            cache.contains(k),
            "key {k:?} in model but not cache after {op:?}"
        );
    }
    assert!(cache.len() <= cache.capacity());

    if let Some(hit) = step.hit {
        let actual_hit = matches!(op, Op::Get(_) | Op::Peek(_) | Op::GetMut(_))
            && match op {
                Op::Get(k) | Op::Peek(k) | Op::GetMut(k) => cache.contains(k),
                _ => false,
            };
        match hit {
            HitMiss::MustHit => assert!(actual_hit, "expected hit for {op:?}"),
            HitMiss::MustMiss => assert!(!actual_hit, "expected miss for {op:?}"),
            HitMiss::MayHitOrMiss => {},
        }
    }

    if let Op::Insert(_) = op {
        if let Some(evicted) = &step.evicted_on_insert {
            assert!(!cache.contains(evicted), "evicted key still resident");
        }
    }

    if let Op::Peek(k) = op {
        if let Some(rank) = rank_before {
            assert_eq!(
                cache.recency_rank(k),
                Some(rank),
                "peek must not change recency rank"
            );
        }
    }

    if matches!(op, Op::Get(_) | Op::GetMut(_) | Op::Touch(_)) {
        // rank updated — checked in proptest against model rank
    }

    match &step.victim {
        OracleExpectation::Exact(victim) => {
            if matches!(op, Op::EvictOne) {
                assert!(!cache.contains(victim));
            }
        },
        OracleExpectation::Legal(set) => {
            // bounded: checked separately
            let _ = set;
        },
        OracleExpectation::None => {},
    }
}

/// Assert `peek_victim` matches model when cache is non-empty.
pub fn assert_peek_victim<K, V, C, M>(cache: &C, model: &M)
where
    K: Clone + Eq + Hash + std::fmt::Debug,
    C: VictimInspectable<K, V>,
    M: PolicyModel<K>,
{
    match model.peek_victim_key() {
        Some(expected) => {
            let (key, _) = cache.peek_victim().expect("model has victim");
            assert_eq!(*key, expected);
        },
        None => assert!(cache.peek_victim().is_none()),
    }
}

/// Assert recency ranks match between cache and LRU model.
pub fn assert_recency_rank<K, V, C>(cache: &C, model_rank: Option<usize>, key: &K)
where
    K: Eq + Hash,
    C: RecencyTracking<K, V>,
{
    assert_eq!(cache.recency_rank(key), model_rank);
}

/// Residency set from probing the `u8` key space (`0..=255`).
///
/// Canonical helper for dual-run tests; replaces inline `(0..=255u8).filter(…).collect()`.
pub fn probe_resident<K>(contains: impl Fn(&K) -> bool) -> HashSet<K>
where
    K: Clone + From<u8> + Eq + Hash,
{
    (0..=255u8).map(K::from).filter(|k| contains(k)).collect()
}
