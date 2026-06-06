//! MRU reference model (victim = most recently used / head).

use std::collections::{HashSet, VecDeque};
use std::hash::Hash;

use crate::abstract_models::{HitMiss, ModelStep, Op, OracleExpectation, PolicyModel};

/// MRU list: head = MRU (eviction victim on insert).
#[derive(Debug, Clone)]
pub struct MruModel<K> {
    order: VecDeque<K>,
    capacity: usize,
}

impl<K> MruModel<K>
where
    K: Clone + Eq + Hash,
{
    pub fn new(capacity: usize) -> Self {
        Self {
            order: VecDeque::new(),
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
        self.order.front().cloned()
    }

    fn promote(&mut self, key: K) {
        self.order.retain(|k| k != &key);
        self.order.push_front(key);
    }
}

impl<K> PolicyModel<K> for MruModel<K>
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
                } else {
                    while self.order.len() >= self.capacity {
                        step.evicted_on_insert = self.order.pop_front();
                    }
                    self.order.push_front(key);
                }
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
                let hit = self.contains_key(&key);
                step.hit = Some(if hit {
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
                    self.order.pop_front();
                }
            },
        }

        step.resident = self.collect_resident();
        step
    }
}
