//! LIFO reference model (victim = newest / top of stack).
//!
//! **Tier:** exact.
//! **Source:** [`docs/testing/specs/lifo.md`](../../../docs/testing/specs/lifo.md) ·
//! [matrix.md](../../../docs/testing/specs/matrix.md)
//! **Cross-model sibling:** [`reference/lifo.rs`](../reference/lifo.rs) (`NaiveLifoModel`).
//! **Tests:** `policy_semantics/lifo_tests.rs` — `VictimInspectable`, `EvictingCache`.
//! **Op strategy:** [`op_strategy`](super::super::op_strategy).

use std::collections::{HashSet, VecDeque};
use std::hash::Hash;

use crate::abstract_models::{HitMiss, ModelStep, Op, OracleExpectation, PolicyModel};

#[derive(Debug, Clone)]
pub struct LifoModel<K> {
    stack: VecDeque<K>,
    capacity: usize,
}

impl<K> LifoModel<K>
where
    K: Clone + Eq + Hash,
{
    pub fn new(capacity: usize) -> Self {
        Self {
            stack: VecDeque::new(),
            capacity,
        }
    }

    fn contains_key(&self, key: &K) -> bool {
        self.stack.iter().any(|k| k == key)
    }

    fn collect_resident(&self) -> HashSet<K> {
        self.stack.iter().cloned().collect()
    }

    fn newest_key(&self) -> Option<K> {
        self.stack.back().cloned()
    }
}

impl<K> PolicyModel<K> for LifoModel<K>
where
    K: Clone + Eq + Hash,
{
    fn capacity(&self) -> usize {
        self.capacity
    }

    fn resident_set(&self) -> HashSet<K> {
        self.collect_resident()
    }

    fn peek_victim_key(&self) -> Option<K> {
        self.newest_key()
    }

    fn apply(&mut self, op: Op<K>) -> ModelStep<K> {
        let mut step = ModelStep::new(self.collect_resident());

        match op {
            Op::Insert(key) => {
                if self.contains_key(&key) {
                    // update — stack order unchanged
                } else if self.capacity == 0 {
                    return step;
                } else {
                    if self.stack.len() >= self.capacity {
                        step.evicted_on_insert = self.stack.pop_back();
                    }
                    self.stack.push_back(key);
                }
            },
            Op::Get(key) | Op::Peek(key) | Op::GetMut(key) => {
                let hit = self.contains_key(&key);
                step.hit = Some(if hit {
                    HitMiss::MustHit
                } else {
                    HitMiss::MustMiss
                });
            },
            Op::Touch(_) => {
                step.hit = Some(HitMiss::MayHitOrMiss);
            },
            Op::Remove(key) => {
                self.stack.retain(|k| k != &key);
            },
            Op::EvictOne => {
                if let Some(victim) = self.newest_key() {
                    step.victim = OracleExpectation::Exact(victim.clone());
                    self.stack.pop_back();
                }
            },
        }

        step.resident = self.collect_resident();
        step
    }
}
