//! Spec-derived Heap-LFU reference model.
//!
//! **Source:** [`docs/testing/specs/heap-lfu.md`](../../../docs/testing/specs/heap-lfu.md) ·
//! [matrix.md](../../../docs/testing/specs/matrix.md)
//! **Tier:** reference (spec-first oracle).
//! **Formulation:** `HashMap<K, u64>` with `Ord` tie-break on key at min frequency;
//! independent of `BinaryHeap` exact model.

use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use crate::abstract_models::{HitMiss, ModelStep, Op, OracleExpectation, PolicyModel};

#[derive(Debug, Clone)]
pub struct NaiveHeapLfuModel<K> {
    freq: HashMap<K, u64>,
    capacity: usize,
}

impl<K> NaiveHeapLfuModel<K>
where
    K: Clone + Eq + Hash + Ord,
{
    pub fn new(capacity: usize) -> Self {
        Self {
            freq: HashMap::new(),
            capacity,
        }
    }

    fn collect_resident(&self) -> HashSet<K> {
        self.freq.keys().cloned().collect()
    }

    fn pick_victim(&self) -> Option<K> {
        self.freq
            .iter()
            .min_by(|(k1, f1), (k2, f2)| f1.cmp(f2).then(k1.cmp(k2)))
            .map(|(k, _)| k.clone())
    }

    fn bump_freq(&mut self, key: &K) {
        if let Some(f) = self.freq.get_mut(key) {
            *f += 1;
        }
    }

    pub fn frequency(&self, key: &K) -> Option<u64> {
        self.freq.get(key).copied()
    }
}

impl<K> PolicyModel<K> for NaiveHeapLfuModel<K>
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
        self.pick_victim()
    }

    fn apply(&mut self, op: Op<K>) -> ModelStep<K> {
        let mut step = ModelStep::new(self.collect_resident());

        match op {
            Op::Insert(key) => {
                if self.freq.contains_key(&key) || self.capacity == 0 {
                    return step;
                }
                if self.freq.len() >= self.capacity {
                    if let Some(victim) = self.pick_victim() {
                        step.evicted_on_insert = Some(victim.clone());
                        self.freq.remove(&victim);
                    }
                }
                self.freq.insert(key, 1);
            },
            Op::Get(key) => {
                let hit = self.freq.contains_key(&key);
                step.hit = Some(if hit {
                    HitMiss::MustHit
                } else {
                    HitMiss::MustMiss
                });
                if hit {
                    self.bump_freq(&key);
                }
            },
            Op::Peek(key) => {
                step.hit = Some(if self.freq.contains_key(&key) {
                    HitMiss::MustHit
                } else {
                    HitMiss::MustMiss
                });
            },
            Op::GetMut(key) => {
                let hit = self.freq.contains_key(&key);
                step.hit = Some(if hit {
                    HitMiss::MustHit
                } else {
                    HitMiss::MustMiss
                });
                if hit {
                    self.bump_freq(&key);
                }
            },
            Op::Touch(key) => {
                if self.freq.contains_key(&key) {
                    self.bump_freq(&key);
                    step.hit = Some(HitMiss::MustHit);
                } else {
                    step.hit = Some(HitMiss::MustMiss);
                }
            },
            Op::Remove(key) => {
                self.freq.remove(&key);
            },
            Op::EvictOne => {
                if let Some(victim) = self.pick_victim() {
                    step.victim = OracleExpectation::Exact(victim.clone());
                    self.freq.remove(&victim);
                }
            },
        }

        step.resident = self.collect_resident();
        step
    }
}
