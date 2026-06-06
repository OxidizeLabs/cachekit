//! Spec-derived FIFO reference model.
//!
//! **Source:** [`docs/testing/specs/fifo.md`](../../../docs/testing/specs/fifo.md) ·
//! [matrix.md](../../../docs/testing/specs/matrix.md)
//! **Formal spec:** [`docs/testing/specs/Fifo.tla`](../../../docs/testing/specs/Fifo.tla) (see
//! [tla-guide.md](../../../docs/testing/specs/tla-guide.md))
//! **Tier:** reference (spec-first oracle).
//! **Formulation:** `HashSet` + append-only `VecDeque` insertion log with stale skips.

use std::collections::{HashSet, VecDeque};
use std::hash::Hash;

use crate::abstract_models::{HitMiss, ModelStep, Op, OracleExpectation, PolicyModel};

#[derive(Debug, Clone)]
pub struct NaiveFifoModel<K> {
    store: HashSet<K>,
    insertion_order: VecDeque<K>,
    capacity: usize,
}

impl<K> NaiveFifoModel<K>
where
    K: Clone + Eq + Hash,
{
    pub fn new(capacity: usize) -> Self {
        Self {
            store: HashSet::new(),
            insertion_order: VecDeque::new(),
            capacity,
        }
    }

    fn collect_resident(&self) -> HashSet<K> {
        self.store.clone()
    }

    fn evict_oldest(&mut self) -> Option<K> {
        while let Some(oldest) = self.insertion_order.pop_front() {
            if self.store.contains(&oldest) {
                self.store.remove(&oldest);
                return Some(oldest);
            }
        }
        None
    }

    fn oldest_key(&self) -> Option<K> {
        self.insertion_order
            .iter()
            .find(|k| self.store.contains(*k))
            .cloned()
    }
}

impl<K> PolicyModel<K> for NaiveFifoModel<K>
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
        self.oldest_key()
    }

    fn apply(&mut self, op: Op<K>) -> ModelStep<K> {
        let mut step = ModelStep::new(self.collect_resident());

        match op {
            Op::Insert(key) => {
                if self.store.contains(&key) {
                    return step;
                }
                if self.capacity == 0 {
                    return step;
                }
                if self.store.len() >= self.capacity {
                    step.evicted_on_insert = self.evict_oldest();
                }
                self.store.insert(key.clone());
                self.insertion_order.push_back(key);
            },
            Op::Get(key) => {
                step.hit = Some(if self.store.contains(&key) {
                    HitMiss::MustHit
                } else {
                    HitMiss::MustMiss
                });
            },
            Op::Peek(key) => {
                step.hit = Some(if self.store.contains(&key) {
                    HitMiss::MustHit
                } else {
                    HitMiss::MustMiss
                });
            },
            Op::GetMut(key) => {
                step.hit = Some(if self.store.contains(&key) {
                    HitMiss::MustHit
                } else {
                    HitMiss::MustMiss
                });
            },
            Op::Touch(_) => {
                step.hit = Some(HitMiss::MayHitOrMiss);
            },
            Op::Remove(key) => {
                self.store.remove(&key);
            },
            Op::EvictOne => {
                if let Some(victim) = self.evict_oldest() {
                    step.victim = OracleExpectation::Exact(victim);
                }
            },
        }

        step.resident = self.collect_resident();
        step
    }
}
