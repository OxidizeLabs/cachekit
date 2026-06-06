//! Policy semantic test harness (abstract interpretation oracles).
//!
//! See [`docs/testing/static-analysis.md`](../../docs/testing/static-analysis.md).

#![allow(dead_code)]

pub mod bounded;
pub mod driver;
pub mod exact;

use std::collections::HashSet;
use std::hash::Hash;

use proptest::prelude::*;

/// Unified trace alphabet for policy semantic tests.
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HitMiss {
    MustHit,
    MustMiss,
    /// Bounded models and TTL partial-knowledge checks only.
    MayHitOrMiss,
}

/// Expected victim from the reference model.
#[derive(Debug, Clone)]
pub enum OracleExpectation<K> {
    Exact(K),
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
pub trait PolicyModel<K> {
    fn capacity(&self) -> usize;
    fn resident_set(&self) -> HashSet<K>;
    fn apply(&mut self, op: Op<K>) -> ModelStep<K>;
    fn peek_victim_key(&self) -> Option<K>;
}

/// Op strategy without `EvictOne` (policies lacking [`EvictingCache`]).
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

/// Shorter traces for O(n) eviction policies (NRU).
pub fn op_strategy_short() -> impl Strategy<Value = Op<u8>> {
    op_strategy()
}

pub fn standard_capacity() -> impl Strategy<Value = usize> {
    1usize..=16
}

pub fn standard_op_list() -> impl Strategy<Value = Vec<Op<u8>>> {
    prop::collection::vec(op_strategy(), 0..120)
}

pub fn standard_op_list_no_evict() -> impl Strategy<Value = Vec<Op<u8>>> {
    prop::collection::vec(op_strategy_no_evict(), 0..120)
}

pub fn short_op_list() -> impl Strategy<Value = Vec<Op<u8>>> {
    prop::collection::vec(op_strategy(), 0..40)
}

pub fn short_op_list_no_evict() -> impl Strategy<Value = Vec<Op<u8>>> {
    prop::collection::vec(op_strategy_no_evict(), 0..40)
}

/// MFU/heap policies: skip `Remove`/`EvictOne` (stale heap vs debug `validate_invariants`).
pub fn op_strategy_mfu_safe() -> impl Strategy<Value = Op<u8>> {
    prop_oneof![
        any::<u8>().prop_map(Op::Insert),
        any::<u8>().prop_map(Op::Get),
        any::<u8>().prop_map(Op::Peek),
        any::<u8>().prop_map(Op::Touch),
    ]
}

pub fn standard_op_list_mfu_safe() -> impl Strategy<Value = Vec<Op<u8>>> {
    prop::collection::vec(op_strategy_mfu_safe(), 0..120)
}

/// Collect resident keys from a cache via iteration pattern.
pub fn resident_from_contains<K, F>(keys: &[K], contains: F) -> HashSet<K>
where
    K: Clone + Eq + Hash,
    F: Fn(&K) -> bool,
{
    keys.iter().filter(|k| contains(k)).cloned().collect()
}
