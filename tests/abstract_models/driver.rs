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

use std::collections::HashSet;
use std::hash::Hash;

use cachekit::traits::{Cache, RecencyTracking, VictimInspectable};

use crate::abstract_models::{ModelStep, Op, PolicyModel};

/// Optional recency rank for cross-model agreement (LRU family).
pub trait ModelRecencyRank<K> {
    /// Recency rank (0 = MRU), or `None` if absent.
    fn model_recency_rank(&self, key: &K) -> Option<usize>;
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

/// Assert cache state matches a [`ModelStep`] after one dual-run op.
///
/// Always checks residency (`probe`), `peek_victim`, and `len <= capacity`.
/// Use [`assert_dual_run_step_no_victim`] when the cache lacks [`VictimInspectable`].
/// Pass an `extra` closure for policy-specific checks (e.g. LFU frequency, LRU recency).
pub fn assert_dual_run_step<K, V, C, M, F>(
    cache: &C,
    model: &M,
    step: &ModelStep<K>,
    probe: impl Fn(&K) -> bool,
    mut extra: F,
) where
    K: Clone + From<u8> + Eq + Hash + std::fmt::Debug,
    C: Cache<K, V> + VictimInspectable<K, V>,
    M: PolicyModel<K>,
    F: FnMut(&C, &M, &ModelStep<K>),
{
    let resident = probe_resident(probe);
    assert_eq!(resident, step.resident);
    assert!(cache.len() <= cache.capacity());
    assert_peek_victim(cache, model);
    extra(cache, model, step);
}

/// Like [`assert_dual_run_step`], but skips `peek_victim` (policies without [`VictimInspectable`]).
pub fn assert_dual_run_step_no_victim<K, V, C, M, F>(
    cache: &C,
    model: &M,
    step: &ModelStep<K>,
    probe: impl Fn(&K) -> bool,
    mut extra: F,
) where
    K: Clone + From<u8> + Eq + Hash + std::fmt::Debug,
    C: Cache<K, V>,
    M: PolicyModel<K>,
    F: FnMut(&C, &M, &ModelStep<K>),
{
    let resident = probe_resident(probe);
    assert_eq!(resident, step.resident);
    assert!(cache.len() <= cache.capacity());
    extra(cache, model, step);
}

/// Apply a trace and run invariant checks after every step (bounded tier).
///
/// The `apply` closure handles policy-specific op mapping (including no-op
/// `GetMut`/`Touch`/`EvictOne`). The `check` closure is caller-supplied — e.g. ARC uses
/// `debug_validate_invariants`; S3-FIFO may wrap `check_invariants` in `#[cfg(debug_assertions)]`.
pub fn run_invariant_trace<K, V, C>(
    cache: &mut C,
    ops: &[Op<K>],
    mut apply: impl FnMut(&mut C, Op<K>),
    check: impl Fn(&C),
) where
    K: Clone,
    C: Cache<K, V>,
{
    for op in ops {
        apply(cache, op.clone());
        assert!(cache.len() <= cache.capacity());
        check(cache);
    }
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

/// Step two [`PolicyModel`] implementations on the same trace and assert agreement.
pub fn assert_models_agree<K, M1, M2>(model_a: &mut M1, model_b: &mut M2, ops: &[Op<K>])
where
    K: Clone + Eq + Hash + std::fmt::Debug,
    M1: PolicyModel<K>,
    M2: PolicyModel<K>,
{
    for op in ops {
        let s1 = model_a.apply(op.clone());
        let s2 = model_b.apply(op.clone());
        assert_eq!(s1.resident, s2.resident, "resident after {op:?}");
        assert_eq!(s1.hit, s2.hit, "hit after {op:?}");
        assert_eq!(
            s1.evicted_on_insert, s2.evicted_on_insert,
            "evicted_on_insert after {op:?}"
        );
        assert_eq!(
            model_a.peek_victim_key(),
            model_b.peek_victim_key(),
            "peek_victim after {op:?}"
        );
    }
}

/// Like [`assert_models_agree`], plus recency rank per resident key (LRU family).
pub fn assert_models_agree_with_recency<K, M1, M2>(
    model_a: &mut M1,
    model_b: &mut M2,
    ops: &[Op<K>],
) where
    K: Clone + Eq + Hash + std::fmt::Debug,
    M1: PolicyModel<K> + ModelRecencyRank<K>,
    M2: PolicyModel<K> + ModelRecencyRank<K>,
{
    for op in ops {
        let s1 = model_a.apply(op.clone());
        let s2 = model_b.apply(op.clone());
        assert_eq!(s1.resident, s2.resident, "resident after {op:?}");
        assert_eq!(s1.hit, s2.hit, "hit after {op:?}");
        assert_eq!(
            s1.evicted_on_insert, s2.evicted_on_insert,
            "evicted_on_insert after {op:?}"
        );
        assert_eq!(
            model_a.peek_victim_key(),
            model_b.peek_victim_key(),
            "peek_victim after {op:?}"
        );
        assert_models_recency_agree(model_a, model_b, &s1.resident, op);
    }
}

/// Recency rank agreement for models implementing [`ModelRecencyRank`].
pub fn assert_models_recency_agree<K, M1, M2>(
    model_a: &M1,
    model_b: &M2,
    resident: &HashSet<K>,
    op: &Op<K>,
) where
    K: Eq + Hash + std::fmt::Debug,
    M1: ModelRecencyRank<K>,
    M2: ModelRecencyRank<K>,
{
    for k in resident {
        assert_eq!(
            model_a.model_recency_rank(k),
            model_b.model_recency_rank(k),
            "recency_rank for {k:?} after {op:?}"
        );
    }
}
