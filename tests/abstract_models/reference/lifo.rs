//! Spec-derived LIFO reference model.
//!
//! **Source:** [`docs/testing/specs/lifo.md`](../../../docs/testing/specs/lifo.md) ·
//! [matrix.md](../../../docs/testing/specs/matrix.md)
//! **Tier:** reference (spec-first oracle).
//! **Formulation:** `Vec<K>` stack (`push`/`pop` end); independent of `VecDeque` exact model.

use std::collections::HashSet;
use std::hash::Hash;

use crate::abstract_models::{HitMiss, ModelStep, Op, OracleExpectation, PolicyModel};

#[derive(Debug, Clone)]
pub struct NaiveLifoModel<K> {
    stack: Vec<K>,
    capacity: usize,
}

impl<K> NaiveLifoModel<K>
where
    K: Clone + Eq + Hash,
{
    pub fn new(capacity: usize) -> Self {
        Self {
            stack: Vec::new(),
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
        self.stack.last().cloned()
    }
}

impl<K> PolicyModel<K> for NaiveLifoModel<K>
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
                    return step;
                }
                if self.capacity == 0 {
                    return step;
                }
                if self.stack.len() >= self.capacity {
                    step.evicted_on_insert = self.stack.pop();
                }
                self.stack.push(key);
            },
            Op::Get(key) | Op::Peek(key) | Op::GetMut(key) => {
                step.hit = Some(if self.contains_key(&key) {
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
                    step.victim = OracleExpectation::Exact(victim);
                    self.stack.pop();
                }
            },
        }

        step.resident = self.collect_resident();
        step
    }
}
