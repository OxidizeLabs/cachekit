//! Spec-derived MRU reference model.
//!
//! **Source:** [`docs/testing/specs/mru.md`](../../../docs/testing/specs/mru.md) ·
//! [matrix.md](../../../docs/testing/specs/matrix.md)
//! **Tier:** reference (spec-first oracle).
//! **Formulation:** `Vec<K>` with index `0` = MRU victim; independent of `VecDeque` exact model.

use std::collections::HashSet;
use std::hash::Hash;

use crate::abstract_models::{HitMiss, ModelStep, Op, OracleExpectation, PolicyModel};

#[derive(Debug, Clone)]
pub struct NaiveMruModel<K> {
    order: Vec<K>,
    capacity: usize,
}

impl<K> NaiveMruModel<K>
where
    K: Clone + Eq + Hash,
{
    pub fn new(capacity: usize) -> Self {
        Self {
            order: Vec::new(),
            capacity,
        }
    }

    fn contains_key(&self, key: &K) -> bool {
        self.order.iter().any(|k| k == key)
    }

    fn collect_resident(&self) -> HashSet<K> {
        self.order.iter().cloned().collect()
    }

    fn mru_key(&self) -> Option<K> {
        self.order.first().cloned()
    }

    fn promote(&mut self, key: K) {
        self.order.retain(|k| k != &key);
        self.order.insert(0, key);
    }
}

impl<K> PolicyModel<K> for NaiveMruModel<K>
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
        self.mru_key()
    }

    fn apply(&mut self, op: Op<K>) -> ModelStep<K> {
        let mut step = ModelStep::new(self.collect_resident());

        match op {
            Op::Insert(key) => {
                if self.contains_key(&key) || self.capacity == 0 {
                    return step;
                }
                while self.order.len() >= self.capacity {
                    step.evicted_on_insert = self.order.first().cloned();
                    if self.order.is_empty() {
                        break;
                    }
                    self.order.remove(0);
                }
                self.order.insert(0, key);
            },
            Op::Get(key) => {
                let hit = self.contains_key(&key);
                step.hit = Some(if hit {
                    HitMiss::MustHit
                } else {
                    HitMiss::MustMiss
                });
                if hit {
                    self.promote(key);
                }
            },
            Op::Peek(key) => {
                step.hit = Some(if self.contains_key(&key) {
                    HitMiss::MustHit
                } else {
                    HitMiss::MustMiss
                });
            },
            Op::GetMut(key) => {
                let hit = self.contains_key(&key);
                step.hit = Some(if hit {
                    HitMiss::MustHit
                } else {
                    HitMiss::MustMiss
                });
                if hit {
                    self.promote(key);
                }
            },
            Op::Touch(_) => {
                step.hit = Some(HitMiss::MayHitOrMiss);
            },
            Op::Remove(key) => {
                self.order.retain(|k| k != &key);
            },
            Op::EvictOne => {
                if let Some(victim) = self.mru_key() {
                    step.victim = OracleExpectation::Exact(victim.clone());
                    self.order.remove(0);
                }
            },
        }

        step.resident = self.collect_resident();
        step
    }
}
