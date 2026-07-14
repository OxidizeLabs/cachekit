//! Spec-derived LFU reference model.
//!
//! **Source:** [`docs/testing/specs/policies/exact/lfu.md`](../../../docs/testing/specs/policies/exact/lfu.md) ·
//! [matrix.md](../../../docs/testing/specs/matrix.md)
//! **Tier:** reference (spec-first oracle).
//! **Formulation:** `HashMap<K, u64>` + append-only bucket-arrival log for FIFO tie-break;
//! independent of [`FrequencyBuckets`](cachekit::ds::FrequencyBuckets).

use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hash;

use crate::abstract_models::{HitMiss, ModelStep, Op, OracleExpectation, PolicyModel};

#[derive(Debug, Clone)]
pub struct NaiveLfuModel<K> {
    freq: HashMap<K, u64>,
    bucket_arrivals: VecDeque<K>,
    capacity: usize,
}

impl<K> NaiveLfuModel<K>
where
    K: Clone + Eq + Hash,
{
    pub fn new(capacity: usize) -> Self {
        Self {
            freq: HashMap::new(),
            bucket_arrivals: VecDeque::new(),
            capacity,
        }
    }

    fn collect_resident(&self) -> HashSet<K> {
        self.freq.keys().cloned().collect()
    }

    fn contains_key(&self, key: &K) -> bool {
        self.freq.contains_key(key)
    }

    fn min_frequency(&self) -> Option<u64> {
        self.freq.values().copied().min()
    }

    /// Last bucket-arrival index for `key` (touches and re-inserts append again).
    fn bucket_arrival_index(&self, key: &K) -> usize {
        self.bucket_arrivals
            .iter()
            .rposition(|k| k == key)
            .unwrap_or(usize::MAX)
    }

    fn peek_victim(&self) -> Option<K> {
        let min = self.min_frequency()?;
        self.freq
            .iter()
            .filter(|(_, f)| **f == min)
            .map(|(k, _)| k)
            .min_by_key(|k| self.bucket_arrival_index(k))
            .cloned()
    }

    fn evict_victim(&mut self) -> Option<K> {
        let victim = self.peek_victim()?;
        self.freq.remove(&victim);
        Some(victim)
    }

    fn touch(&mut self, key: &K) -> bool {
        if let Some(f) = self.freq.get_mut(key) {
            *f = f.saturating_add(1);
            self.bucket_arrivals.push_back(key.clone());
            true
        } else {
            false
        }
    }

    fn insert_new(&mut self, key: K) -> Option<K> {
        if self.capacity == 0 {
            return None;
        }
        let mut evicted = None;
        if self.freq.len() >= self.capacity {
            evicted = self.evict_victim();
        }
        self.freq.insert(key.clone(), 1);
        self.bucket_arrivals.push_back(key);
        evicted
    }

    pub fn frequency(&self, key: &K) -> Option<u64> {
        self.freq.get(key).copied()
    }
}

impl<K> PolicyModel<K> for NaiveLfuModel<K>
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
        self.peek_victim()
    }

    fn apply(&mut self, op: Op<K>) -> ModelStep<K> {
        let mut step = ModelStep::new(self.collect_resident());

        match op {
            Op::Insert(key) => {
                if self.contains_key(&key) {
                    return step;
                }
                step.evicted_on_insert = self.insert_new(key);
            },
            Op::Get(key) => {
                let hit = self.contains_key(&key);
                step.hit = Some(if hit {
                    HitMiss::MustHit
                } else {
                    HitMiss::MustMiss
                });
                if hit {
                    self.touch(&key);
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
                    self.touch(&key);
                }
            },
            Op::Touch(key) => {
                if self.touch(&key) {
                    step.hit = Some(HitMiss::MustHit);
                } else {
                    step.hit = Some(HitMiss::MustMiss);
                }
            },
            Op::Remove(key) => {
                self.freq.remove(&key);
            },
            Op::EvictOne => {
                if let Some(victim) = self.evict_victim() {
                    step.victim = OracleExpectation::Exact(victim);
                }
            },
        }

        step.resident = self.collect_resident();
        step
    }
}
