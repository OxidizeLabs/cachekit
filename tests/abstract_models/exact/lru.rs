//! LRU occupancy model (MRU at front, LRU at back).
//!
//! **Tier:** exact.
//! **Source:** [`docs/testing/specs/lru.md`](../../../docs/testing/specs/lru.md) ·
//! [fast-lru.md](../../../docs/testing/specs/fast-lru.md) ·
//! [matrix.md](../../../docs/testing/specs/matrix.md)
//! **Recency:** rank 0 = MRU; used by `assert_recency_rank` in LRU and Fast-LRU tests.
//! **Tests:** `policy_semantics/lru_tests.rs`, `fast_lru_tests.rs`; also composed in
//! `ttl_integration_test.rs`.
//! **Op strategy:** [`op_strategy`](super::super::op_strategy) (LRU);
//! [`op_strategy_with_get_mut`](super::super::op_strategy_with_get_mut) (Fast-LRU).

use std::collections::{HashSet, VecDeque};
use std::hash::Hash;

use crate::abstract_models::driver::ModelRecencyRank;
use crate::abstract_models::{HitMiss, ModelStep, Op, OracleExpectation, PolicyModel};

/// MRU-first deque matching `FastLru` head/tail semantics.
#[derive(Debug, Clone)]
pub struct LruOccupancyModel<K> {
    order: VecDeque<K>,
    capacity: usize,
}

impl<K> LruOccupancyModel<K>
where
    K: Clone + Eq + Hash,
{
    pub fn new(capacity: usize) -> Self {
        Self {
            order: VecDeque::new(),
            capacity,
        }
    }

    pub fn touch_key(&mut self, key: K) {
        self.order.retain(|k| k != &key);
        self.order.push_front(key);
    }

    fn contains_key(&self, key: &K) -> bool {
        self.order.iter().any(|k| k == key)
    }

    fn collect_resident(&self) -> HashSet<K> {
        self.order.iter().cloned().collect()
    }

    fn lru_key(&self) -> Option<K> {
        self.order.back().cloned()
    }

    fn recency_rank(&self, key: &K) -> Option<usize> {
        self.order.iter().position(|k| k == key)
    }

    fn insert_new(&mut self, key: K) -> Option<K> {
        if self.capacity == 0 {
            return None;
        }
        let mut evicted = None;
        while self.order.len() >= self.capacity {
            evicted = self.order.pop_back();
        }
        self.order.push_front(key);
        evicted
    }
}

impl<K> PolicyModel<K> for LruOccupancyModel<K>
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
        self.lru_key()
    }

    fn apply(&mut self, op: Op<K>) -> ModelStep<K> {
        let mut step = ModelStep::new(self.collect_resident());

        match op {
            Op::Insert(key) => {
                if self.contains_key(&key) {
                    self.touch_key(key);
                } else {
                    step.evicted_on_insert = self.insert_new(key);
                }
                step.victim = OracleExpectation::None;
            },
            Op::Get(key) => {
                let hit = self.contains_key(&key);
                step.hit = Some(if hit {
                    HitMiss::MustHit
                } else {
                    HitMiss::MustMiss
                });
                if hit {
                    self.touch_key(key);
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
                    self.touch_key(key);
                }
            },
            Op::Touch(key) => {
                if self.contains_key(&key) {
                    self.touch_key(key);
                    step.hit = Some(HitMiss::MustHit);
                } else {
                    step.hit = Some(HitMiss::MustMiss);
                }
            },
            Op::Remove(key) => {
                if self.contains_key(&key) {
                    self.order.retain(|k| k != &key);
                }
            },
            Op::EvictOne => {
                if let Some(victim) = self.lru_key() {
                    step.victim = OracleExpectation::Exact(victim.clone());
                    self.order.pop_back();
                }
            },
        }

        step.resident = self.collect_resident();
        step
    }
}

impl<K> ModelRecencyRank<K> for LruOccupancyModel<K>
where
    K: Clone + Eq + Hash,
{
    fn model_recency_rank(&self, key: &K) -> Option<usize> {
        self.recency_rank(key)
    }
}

impl<K> LruOccupancyModel<K>
where
    K: Clone + Eq + Hash,
{
    /// Recency rank for assertions (0 = MRU).
    pub fn model_recency_rank(&self, key: &K) -> Option<usize> {
        self.recency_rank(key)
    }
}
