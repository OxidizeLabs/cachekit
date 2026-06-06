//! Spec-derived LRU reference model (timestamp formulation).
//!
//! **Source:** [`docs/testing/specs/policies/exact/lru.md`](../../../docs/testing/specs/policies/exact/lru.md) ·
//! [matrix.md](../../../docs/testing/specs/matrix.md)
//! **Tier:** reference (spec-first oracle).
//! **Formulation:** `HashMap<K, u64>` access times + monotonic clock; independent of deque model.

use std::collections::HashMap;
use std::collections::HashSet;
use std::hash::Hash;

use crate::abstract_models::driver::ModelRecencyRank;
use crate::abstract_models::{HitMiss, ModelStep, Op, OracleExpectation, PolicyModel};

#[derive(Debug, Clone)]
pub struct NaiveLruModel<K> {
    access: HashMap<K, u64>,
    clock: u64,
    capacity: usize,
}

impl<K> NaiveLruModel<K>
where
    K: Clone + Eq + Hash + Ord,
{
    pub fn new(capacity: usize) -> Self {
        Self {
            access: HashMap::new(),
            clock: 0,
            capacity,
        }
    }

    fn collect_resident(&self) -> HashSet<K> {
        self.access.keys().cloned().collect()
    }

    fn bump_clock(&mut self) -> u64 {
        self.clock = self.clock.saturating_add(1);
        self.clock
    }

    fn promote(&mut self, key: K) {
        let ts = self.bump_clock();
        self.access.insert(key, ts);
    }

    fn lru_victim(&self) -> Option<K> {
        self.access
            .iter()
            .min_by(|(k1, t1), (k2, t2)| t1.cmp(t2).then_with(|| k1.cmp(k2)))
            .map(|(k, _)| k.clone())
    }

    fn evict_lru(&mut self) -> Option<K> {
        let victim = self.lru_victim()?;
        self.access.remove(&victim);
        Some(victim)
    }

    fn insert_new(&mut self, key: K) -> Option<K> {
        if self.capacity == 0 {
            return None;
        }
        let mut evicted = None;
        while self.access.len() >= self.capacity {
            evicted = self.evict_lru();
        }
        self.promote(key);
        evicted
    }

    fn sorted_resident_keys(&self) -> Vec<K> {
        let mut keys: Vec<K> = self.access.keys().cloned().collect();
        keys.sort_by(|a, b| {
            let ta = self.access[a];
            let tb = self.access[b];
            tb.cmp(&ta).then_with(|| a.cmp(b))
        });
        keys
    }
}

impl<K> ModelRecencyRank<K> for NaiveLruModel<K>
where
    K: Clone + Eq + Hash + Ord,
{
    fn model_recency_rank(&self, key: &K) -> Option<usize> {
        self.sorted_resident_keys().iter().position(|k| k == key)
    }
}

impl<K> NaiveLruModel<K>
where
    K: Clone + Eq + Hash + Ord,
{
    /// Recency rank for assertions (0 = MRU).
    pub fn model_recency_rank(&self, key: &K) -> Option<usize> {
        self.sorted_resident_keys().iter().position(|k| k == key)
    }
}

impl<K> PolicyModel<K> for NaiveLruModel<K>
where
    K: Clone + Eq + Hash + Ord,
{
    fn capacity(&self) -> usize {
        self.capacity
    }

    fn resident_set(&self) -> HashSet<K> {
        self.collect_resident()
    }

    fn peek_victim_key(&self) -> Option<K> {
        self.lru_victim()
    }

    fn apply(&mut self, op: Op<K>) -> ModelStep<K> {
        let mut step = ModelStep::new(self.collect_resident());

        match op {
            Op::Insert(key) => {
                if self.access.contains_key(&key) {
                    self.promote(key);
                } else {
                    step.evicted_on_insert = self.insert_new(key);
                }
            },
            Op::Get(key) => {
                if self.access.contains_key(&key) {
                    step.hit = Some(HitMiss::MustHit);
                    self.promote(key);
                } else {
                    step.hit = Some(HitMiss::MustMiss);
                }
            },
            Op::Peek(key) => {
                step.hit = Some(if self.access.contains_key(&key) {
                    HitMiss::MustHit
                } else {
                    HitMiss::MustMiss
                });
            },
            Op::GetMut(key) => {
                if self.access.contains_key(&key) {
                    step.hit = Some(HitMiss::MustHit);
                    self.promote(key);
                } else {
                    step.hit = Some(HitMiss::MustMiss);
                }
            },
            Op::Touch(key) => {
                if self.access.contains_key(&key) {
                    step.hit = Some(HitMiss::MustHit);
                    self.promote(key);
                } else {
                    step.hit = Some(HitMiss::MustMiss);
                }
            },
            Op::Remove(key) => {
                self.access.remove(&key);
            },
            Op::EvictOne => {
                if let Some(victim) = self.evict_lru() {
                    step.victim = OracleExpectation::Exact(victim);
                }
            },
        }

        step.resident = self.collect_resident();
        step
    }
}
