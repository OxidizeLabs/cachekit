//! Policy semantic test harness (abstract interpretation oracles).
//!
//! ## Architecture
//!
//! Reference models predict cache observables from access traces. Integration tests in
//! [`policy_semantics`](../policy_semantics/) dual-run each [`PolicyModel`] against the real
//! policy implementation step by step.
//!
//! ```text
//! Op trace ──► PolicyModel::apply ──► ModelStep ──► assert vs cache
//! ```
//!
//! Models live under [`exact`] (deterministic victims) and [`bounded`] (doc stubs).
//! Submodules and op strategies are gated by matching `policy-*` features.
//! Assertion helpers are in [`driver`].
//!
//! ## Key components
//!
//! - [`Op`] — unified trace alphabet (`Insert`, `Get`, `Peek`, `GetMut`, `Touch`, `Remove`, `EvictOne`)
//! - [`HitMiss`] — `MustHit` / `MustMiss` / `MayHitOrMiss` (bounded and TTL only)
//! - [`ModelStep`] — residency, hit classification, victim expectation, insert eviction
//! - [`OracleExpectation`] — `Exact(key)`, `Legal(set)`, or `None`
//! - [`PolicyModel`] — `apply`, `peek_victim_key`, `resident_set`, `capacity`
//!
//! ## Proptest strategies
//!
//! Use [`op_strategy_no_evict`] for policies without [`EvictingCache`](../../src/traits.rs).
//! Use [`op_strategy_with_get_mut`] for Fast-LRU and S3-FIFO. Use [`op_strategy_mfu_safe`] when
//! `Remove`/`EvictOne` would leave a stale heap (MFU, Heap-LFU).
//!
//! ## Further reading
//!
//! - [README](README.md) — directory layout, policy matrix, contributor checklist
//! - [Policy semantic testing](../../docs/testing/static-analysis.md) — full harness design and CI
//!
//! ## Multi-crate usage
//!
//! `#[path]`-included by `policy_semantics` (full matrix) and `ttl_integration_test` (LRU
//! subset). Each integration-test binary uses a different subset of models and helpers.
#![allow(dead_code)]

#[cfg(any(
    feature = "policy-arc",
    feature = "policy-car",
    feature = "policy-clock-pro",
    feature = "policy-s3-fifo"
))]
pub mod bounded;
pub mod driver;
pub mod exact;

use std::collections::HashSet;
use std::hash::Hash;

use proptest::prelude::*;

/// Unified trace alphabet for policy semantic tests.
///
/// Maps to cache API calls in each `policy_semantics/*_tests.rs` adapter. `Peek` must not
/// promote recency; `Get`, `GetMut`, and `Touch` do on LRU-family policies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op<K> {
    Insert(K),
    Get(K),
    Peek(K),
    GetMut(K),
    Touch(K),
    Remove(K),
    EvictOne,
}

/// Hit/miss classification for the current operation.
///
/// Exact models use `MustHit` / `MustMiss`. Bounded models and TTL checks may use
/// `MayHitOrMiss` when knowledge is partial.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HitMiss {
    MustHit,
    MustMiss,
    /// Bounded models and TTL partial-knowledge checks only.
    MayHitOrMiss,
}

/// Expected victim from the reference model.
///
/// `Exact` — deterministic victim (exact-tier models). `Legal` — any resident key is
/// admissible (bounded-tier). `None` — no victim expectation for this step.
#[derive(Debug, Clone)]
pub enum OracleExpectation<K> {
    Exact(K),
    /// Reserved for future bounded-tier legal victim sets.
    Legal(HashSet<K>),
    None,
}

impl<K: Eq + Hash> PartialEq for OracleExpectation<K> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Exact(a), Self::Exact(b)) => a == b,
            (Self::Legal(a), Self::Legal(b)) => a == b,
            (Self::None, Self::None) => true,
            _ => false,
        }
    }
}

impl<K: Eq + Hash> Eq for OracleExpectation<K> {}

/// Observables produced by applying one op to the reference model.
///
/// Dual-run tests compare each field against the real cache after the same op.
#[derive(Debug, Clone)]
pub struct ModelStep<K> {
    pub resident: HashSet<K>,
    pub hit: Option<HitMiss>,
    pub victim: OracleExpectation<K>,
    pub evicted_on_insert: Option<K>,
}

impl<K> ModelStep<K> {
    pub fn new(resident: HashSet<K>) -> Self {
        Self {
            resident,
            hit: None,
            victim: OracleExpectation::None,
            evicted_on_insert: None,
        }
    }
}

/// Reference semantics for a cache policy.
///
/// Each implementation encodes one policy's eviction rule. See [`exact`] and [`bounded`] modules
/// and the [README](README.md) for the per-policy model matrix.
pub trait PolicyModel<K> {
    fn capacity(&self) -> usize;
    fn resident_set(&self) -> HashSet<K>;
    fn apply(&mut self, op: Op<K>) -> ModelStep<K>;
    fn peek_victim_key(&self) -> Option<K>;
}

/// Op strategy without `EvictOne` (policies lacking [`EvictingCache`]).
#[cfg(any(
    feature = "policy-two-q",
    feature = "policy-slru",
    feature = "policy-nru"
))]
pub fn op_strategy_no_evict() -> impl Strategy<Value = Op<u8>> {
    prop_oneof![
        any::<u8>().prop_map(Op::Insert),
        any::<u8>().prop_map(Op::Get),
        any::<u8>().prop_map(Op::Peek),
        any::<u8>().prop_map(Op::Touch),
        any::<u8>().prop_map(Op::Remove),
    ]
}

/// Default op strategy for policies without `GetMut`.
pub fn op_strategy() -> impl Strategy<Value = Op<u8>> {
    prop_oneof![
        any::<u8>().prop_map(Op::Insert),
        any::<u8>().prop_map(Op::Get),
        any::<u8>().prop_map(Op::Peek),
        any::<u8>().prop_map(Op::Touch),
        any::<u8>().prop_map(Op::Remove),
        Just(Op::EvictOne),
    ]
}

/// Op strategy including `GetMut` (Fast-LRU, S3-FIFO).
#[cfg(any(feature = "policy-fast-lru", feature = "policy-s3-fifo"))]
pub fn op_strategy_with_get_mut() -> impl Strategy<Value = Op<u8>> {
    prop_oneof![
        6 => any::<u8>().prop_map(Op::Insert),
        4 => any::<u8>().prop_map(Op::Get),
        2 => any::<u8>().prop_map(Op::Peek),
        2 => any::<u8>().prop_map(Op::GetMut),
        2 => any::<u8>().prop_map(Op::Touch),
        2 => any::<u8>().prop_map(Op::Remove),
        1 => Just(Op::EvictOne),
    ]
}

pub fn standard_capacity() -> impl Strategy<Value = usize> {
    1usize..=16
}

pub fn standard_op_list() -> impl Strategy<Value = Vec<Op<u8>>> {
    prop::collection::vec(op_strategy(), 0..120)
}

#[cfg(any(feature = "policy-two-q", feature = "policy-slru"))]
pub fn standard_op_list_no_evict() -> impl Strategy<Value = Vec<Op<u8>>> {
    prop::collection::vec(op_strategy_no_evict(), 0..120)
}

#[cfg(feature = "policy-nru")]
pub fn short_op_list_no_evict() -> impl Strategy<Value = Vec<Op<u8>>> {
    prop::collection::vec(op_strategy_no_evict(), 0..40)
}

/// MFU: skip `Remove`/`EvictOne` (stale heap vs debug `validate_invariants`).
#[cfg(feature = "policy-mfu")]
pub fn op_strategy_mfu_safe() -> impl Strategy<Value = Op<u8>> {
    prop_oneof![
        any::<u8>().prop_map(Op::Insert),
        any::<u8>().prop_map(Op::Get),
        any::<u8>().prop_map(Op::Peek),
        any::<u8>().prop_map(Op::Touch),
    ]
}

#[cfg(feature = "policy-mfu")]
pub fn standard_op_list_mfu_safe() -> impl Strategy<Value = Vec<Op<u8>>> {
    prop::collection::vec(op_strategy_mfu_safe(), 0..120)
}
